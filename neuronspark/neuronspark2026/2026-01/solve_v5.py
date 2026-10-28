"""
NS-2026-01 V5 - 单模型极速版 (目标<60s)
更大lr, 限1500轮, 重点靠特征提分
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# 加载
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
calendar = pd.read_csv('calendar.csv')
weather = pd.read_csv('weather.csv')
menu = pd.read_csv('menu.csv')

def merge_features(df):
    df = df.merge(calendar, on='date', how='left')
    df = df.merge(weather, on=['date', 'meal'], how='left')
    df = df.merge(menu, on=['date', 'meal', 'canteen_area'], how='left')
    return df

train = merge_features(train)
test = merge_features(test)

# 特征
def build_features(df):
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date_dt'].dt.dayofweek
    df['day_of_month'] = df['date_dt'].dt.day
    df['month'] = df['date_dt'].dt.month
    df['day_of_year'] = df['date_dt'].dt.dayofyear
    df['week_of_year'] = df['date_dt'].dt.isocalendar().week.astype(int)
    
    meal_map = {'breakfast': 0, 'lunch': 1, 'dinner': 2}
    df['meal_enc'] = df['meal'].map(meal_map)
    area_map = {f'A0{i}': i for i in range(1, 7)}
    df['area_enc'] = df['canteen_area'].map(area_map)
    weather_map = {'clear': 0, 'cloudy': 1, 'rain': 2}
    df['weather_enc'] = df['weather'].map(weather_map)
    menu_type_map = {'noodle': 0, 'porridge': 1, 'light': 2, 'spicy': 3, 
                     'special': 4, 'rice_set': 5, 'western': 6}
    df['menu_type_enc'] = df['menu_type'].map(menu_type_map)
    
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['meal_area'] = df['meal_enc'] * 10 + df['area_enc']
    df['weekday_meal'] = df['day_of_week'] * 3 + df['meal_enc']
    df['weekday_area'] = df['day_of_week'] * 6 + df['area_enc']
    df['weekday_meal_area'] = df['day_of_week'] * 18 + df['meal_enc'] * 6 + df['area_enc']
    df['is_workday'] = ((df['is_weekend'] == 0) & (df['is_holiday'] == 0)).astype(int)
    
    df['di_x_pop'] = df['demand_index'] * df['menu_popularity']
    df['di_x_promo'] = df['demand_index'] * df['is_promotion']
    df['di_x_area'] = df['demand_index'] * df['area_enc']
    df['di_x_meal'] = df['demand_index'] * df['meal_enc']
    df['semester_progress'] = df['semester_week'] / 22.0
    df['exam_meal'] = df['is_exam_week'] * df['meal_enc']
    df['holiday_meal'] = df['is_holiday'] * df['meal_enc']
    
    return df

train = build_features(train)
test = build_features(test)
train_sorted = train.sort_values('date_dt').reset_index(drop=True)

# 统计特征
for gcols, suffix in [(['meal_enc','area_enc','day_of_week'], 'maw'), (['meal_enc','area_enc'], 'ma')]:
    stats = train_sorted.groupby(gcols)['volume'].agg(['mean','median','std']).reset_index()
    stats.columns = gcols + [f'{suffix}_mean', f'{suffix}_median', f'{suffix}_std']
    train = train.merge(stats, on=gcols, how='left')
    test = test.merge(stats, on=gcols, how='left')

for weeks in [2, 4, 8]:
    cutoff = train_sorted['date_dt'].max() - pd.Timedelta(weeks=weeks)
    recent = train_sorted[train_sorted['date_dt'] >= cutoff]
    rs = recent.groupby(['meal_enc','area_enc','day_of_week'])['volume'].mean().reset_index()
    rs.columns = ['meal_enc','area_enc','day_of_week', f'r{weeks}w_mean']
    train = train.merge(rs, on=['meal_enc','area_enc','day_of_week'], how='left')
    test = test.merge(rs, on=['meal_enc','area_enc','day_of_week'], how='left')

di_stats = train_sorted.groupby(['meal_enc','day_of_week'])['demand_index'].mean().reset_index()
di_stats.columns = ['meal_enc','day_of_week','di_hist_mean']
train = train.merge(di_stats, on=['meal_enc','day_of_week'], how='left')
test = test.merge(di_stats, on=['meal_enc','day_of_week'], how='left')
train['di_dev'] = train['demand_index'] / train['di_hist_mean'].clip(lower=1)
test['di_dev'] = test['demand_index'] / test['di_hist_mean'].clip(lower=1)

feature_cols = [
    'meal_enc', 'area_enc', 'day_of_week', 'day_of_month', 'month',
    'week_of_year', 'day_of_year', 'weekday', 'is_weekend', 'semester_week',
    'is_holiday', 'is_exam_week', 'is_makeup_day', 'campus_event_level',
    'demand_index', 'temperature', 'feels_like', 'humidity', 'wind_speed',
    'rain_level', 'weather_enc', 'menu_type_enc', 'menu_popularity', 'is_promotion',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'meal_area', 'weekday_meal', 'weekday_area', 'weekday_meal_area', 'is_workday',
    'di_x_pop', 'di_x_promo', 'di_x_area', 'di_x_meal',
    'semester_progress', 'exam_meal', 'holiday_meal', 'di_dev', 'di_hist_mean',
]
stat_cols = [c for c in train.columns if any(c.startswith(p) for p in ['maw_','ma_','r2w_','r4w_','r8w_'])]
feature_cols += stat_cols

X_all = train[feature_cols].values.astype(np.float32)
y_all = train['volume'].values
X_test = test[feature_cols].values.astype(np.float32)

# 验证
train['date_dt'] = pd.to_datetime(train['date'])
val_start = train['date_dt'].max() - pd.Timedelta(days=89)
val_mask = (train['date_dt'] >= val_start).values
tr_mask = ~val_mask
X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
X_val, y_val = X_all[val_mask], y_all[val_mask]

print(f"Features: {len(feature_cols)}, Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}")

# 单模型 - 高lr快速收敛
params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.08,
    'num_leaves': 127,
    'min_child_samples': 15,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.05,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

dtrain = lgb.Dataset(X_tr, label=y_tr)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
model = lgb.train(params, dtrain, num_boost_round=1500,
                  valid_sets=[dtrain, dval],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(500)])

val_pred = np.clip(model.predict(X_val), 0, None)

mae = np.mean(np.abs(y_val - val_pred))
rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
mape = np.mean(np.abs((y_val - val_pred) / np.clip(y_val, 1, None)))
s = 200*max(0,1-mae/1000) + 150*max(0,1-rmse/1500) + 150*max(0,1-mape/0.5)
print(f"\nValidation: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.4f}, Score={s:.1f}/500")
print(f"Best iteration: {model.best_iteration}")

# Full retrain
dtrain_full = lgb.Dataset(X_all, label=y_all)
model_full = lgb.train(params, dtrain_full, num_boost_round=model.best_iteration)
test_pred = np.clip(model_full.predict(X_test), 0, None)

# 提交
submission = test[['date', 'meal', 'canteen_area']].copy()
submission['volume'] = np.round(test_pred, 2)
submission.to_csv('results.csv', index=False)
print(f"Submission: {submission.shape}, range=[{submission['volume'].min():.1f}, {submission['volume'].max():.1f}]")
