"""
NS-2026-01 智慧饭堂：人流与备餐预测
LightGBM + 特征工程方案
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ============ 1. 加载数据 ============
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
calendar = pd.read_csv('calendar.csv')
weather = pd.read_csv('weather.csv')
menu = pd.read_csv('menu.csv')

# ============ 2. 合并特征 ============
def merge_features(df):
    # Merge calendar (on date)
    df = df.merge(calendar, on='date', how='left')
    # Merge weather (on date + meal)
    df = df.merge(weather, on=['date', 'meal'], how='left')
    # Merge menu (on date + meal + canteen_area)
    df = df.merge(menu, on=['date', 'meal', 'canteen_area'], how='left')
    return df

train = merge_features(train)
test = merge_features(test)

# ============ 3. 特征工程 ============
def build_features(df):
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'])
    
    # 时间特征
    df['day_of_week'] = df['date_dt'].dt.dayofweek
    df['day_of_month'] = df['date_dt'].dt.day
    df['month'] = df['date_dt'].dt.month
    df['day_of_year'] = df['date_dt'].dt.dayofyear
    df['week_of_year'] = df['date_dt'].dt.isocalendar().week.astype(int)
    
    # 编码 meal
    meal_map = {'breakfast': 0, 'lunch': 1, 'dinner': 2}
    df['meal_enc'] = df['meal'].map(meal_map)
    
    # 编码 canteen_area
    area_map = {f'A0{i}': i for i in range(1, 7)}
    df['area_enc'] = df['canteen_area'].map(area_map)
    
    # 编码 weather type
    weather_map = {'clear': 0, 'cloudy': 1, 'rain': 2}
    df['weather_enc'] = df['weather'].map(weather_map)
    
    # 编码 menu_type
    menu_type_map = {'noodle': 0, 'porridge': 1, 'light': 2, 'spicy': 3, 
                     'special': 4, 'rice_set': 5, 'western': 6}
    df['menu_type_enc'] = df['menu_type'].map(menu_type_map)
    
    # 周期性特征 (sin/cos)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 交互特征
    df['meal_area'] = df['meal_enc'] * 10 + df['area_enc']
    df['weekday_meal'] = df['day_of_week'] * 3 + df['meal_enc']
    df['weekday_area'] = df['day_of_week'] * 6 + df['area_enc']
    
    # 是否工作日
    df['is_workday'] = ((df['is_weekend'] == 0) & (df['is_holiday'] == 0)).astype(int)
    
    return df

train = build_features(train)
test = build_features(test)

# ============ 4. 历史统计特征 ============
# 按 (meal, canteen_area, day_of_week) 的均值/中位数
train_sorted = train.sort_values('date_dt').reset_index(drop=True)

# 按组统计 (用全部训练集的统计作为特征)
group_stats = train.groupby(['meal_enc', 'area_enc', 'day_of_week'])['volume'].agg(
    ['mean', 'median', 'std']).reset_index()
group_stats.columns = ['meal_enc', 'area_enc', 'day_of_week', 'grp_mean', 'grp_median', 'grp_std']

train = train.merge(group_stats, on=['meal_enc', 'area_enc', 'day_of_week'], how='left')
test = test.merge(group_stats, on=['meal_enc', 'area_enc', 'day_of_week'], how='left')

# 按 (meal, canteen_area) 整体均值
area_meal_stats = train_sorted.groupby(['meal_enc', 'area_enc'])['volume'].agg(
    ['mean', 'std']).reset_index()
area_meal_stats.columns = ['meal_enc', 'area_enc', 'area_meal_mean', 'area_meal_std']

train = train.merge(area_meal_stats, on=['meal_enc', 'area_enc'], how='left')
test = test.merge(area_meal_stats, on=['meal_enc', 'area_enc'], how='left')

# 最近N周同(meal, area, weekday)的均值 - 用训练集最后几周
last_weeks = train_sorted[train_sorted['date_dt'] >= train_sorted['date_dt'].max() - pd.Timedelta(weeks=4)]
recent_stats = last_weeks.groupby(['meal_enc', 'area_enc', 'day_of_week'])['volume'].mean().reset_index()
recent_stats.columns = ['meal_enc', 'area_enc', 'day_of_week', 'recent_4w_mean']

train = train.merge(recent_stats, on=['meal_enc', 'area_enc', 'day_of_week'], how='left')
test = test.merge(recent_stats, on=['meal_enc', 'area_enc', 'day_of_week'], how='left')

# 最近2周
last_2w = train_sorted[train_sorted['date_dt'] >= train_sorted['date_dt'].max() - pd.Timedelta(weeks=2)]
recent_2w = last_2w.groupby(['meal_enc', 'area_enc', 'day_of_week'])['volume'].mean().reset_index()
recent_2w.columns = ['meal_enc', 'area_enc', 'day_of_week', 'recent_2w_mean']

train = train.merge(recent_2w, on=['meal_enc', 'area_enc', 'day_of_week'], how='left')
test = test.merge(recent_2w, on=['meal_enc', 'area_enc', 'day_of_week'], how='left')

# ============ 5. 定义特征列 ============
feature_cols = [
    'meal_enc', 'area_enc', 'day_of_week', 'day_of_month', 'month',
    'week_of_year', 'day_of_year',
    'weekday', 'is_weekend', 'semester_week', 'is_holiday', 'is_exam_week',
    'is_makeup_day', 'campus_event_level',
    'demand_index', 'temperature', 'feels_like', 'humidity', 'wind_speed',
    'rain_level', 'weather_enc',
    'menu_type_enc', 'menu_popularity', 'is_promotion',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'meal_area', 'weekday_meal', 'weekday_area', 'is_workday',
    'grp_mean', 'grp_median', 'grp_std',
    'area_meal_mean', 'area_meal_std',
    'recent_4w_mean', 'recent_2w_mean',
]

X_train = train[feature_cols].values
y_train = train['volume'].values
X_test = test[feature_cols].values

print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
print(f"Test samples: {X_test.shape[0]}")

# ============ 6. 时间序列验证 ============
# 用最后90天作为验证集（模拟测试集长度）
train['date_dt'] = pd.to_datetime(train['date'])
val_start = train['date_dt'].max() - pd.Timedelta(days=89)
val_mask = train['date_dt'] >= val_start
train_mask = ~val_mask

X_tr = X_train[train_mask]
y_tr = y_train[train_mask]
X_val = X_train[val_mask]
y_val = y_train[val_mask]

print(f"\nValidation split: train={X_tr.shape[0]}, val={X_val.shape[0]}")

# ============ 7. LightGBM 训练 ============
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

# Train on validation split first to check performance
dtrain = lgb.Dataset(X_tr, label=y_tr)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

model_val = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dtrain, dval],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)]
)

# Evaluate
val_pred = model_val.predict(X_val)
val_pred = np.clip(val_pred, 0, None)

mae = np.mean(np.abs(y_val - val_pred))
rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
mape = np.mean(np.abs((y_val - val_pred) / np.clip(y_val, 1, None)))

mae_score = 200 * max(0, 1 - mae / 1000.0)
rmse_score = 150 * max(0, 1 - rmse / 1500.0)
mape_score = 150 * max(0, 1 - mape / 0.50)
total_score = mae_score + rmse_score + mape_score

print(f"\n=== Validation Results ===")
print(f"MAE:  {mae:.2f} -> score: {mae_score:.1f}/200")
print(f"RMSE: {rmse:.2f} -> score: {rmse_score:.1f}/150")
print(f"MAPE: {mape:.4f} -> score: {mape_score:.1f}/150")
print(f"Total: {total_score:.1f}/500")

# ============ 8. 最终模型：用全部训练数据 ============
dtrain_full = lgb.Dataset(X_train, label=y_train)
model_final = lgb.train(
    params,
    dtrain_full,
    num_boost_round=model_val.best_iteration,
)

# ============ 9. 预测 & 生成提交 ============
test_pred = model_final.predict(X_test)
test_pred = np.clip(test_pred, 0, None)

submission = test[['date', 'meal', 'canteen_area']].copy()
submission['volume'] = np.round(test_pred, 2)
submission.to_csv('results.csv', index=False)

print(f"\n=== Submission Generated ===")
print(f"File: results.csv")
print(f"Shape: {submission.shape}")
print(f"Volume range: [{submission['volume'].min():.2f}, {submission['volume'].max():.2f}]")
print(f"Volume mean: {submission['volume'].mean():.2f}")
print(submission.head(10))
