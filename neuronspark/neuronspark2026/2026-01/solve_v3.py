"""
NS-2026-01 智慧饭堂 - V3 快速多模型融合
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============ 加载 ============
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

# ============ 特征工程 ============
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
    
    # 周期编码
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 交互
    df['meal_area'] = df['meal_enc'] * 10 + df['area_enc']
    df['weekday_meal'] = df['day_of_week'] * 3 + df['meal_enc']
    df['weekday_area'] = df['day_of_week'] * 6 + df['area_enc']
    df['weekday_meal_area'] = df['day_of_week'] * 18 + df['meal_enc'] * 6 + df['area_enc']
    df['is_workday'] = ((df['is_weekend'] == 0) & (df['is_holiday'] == 0)).astype(int)
    
    # demand_index 交互
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

# ============ 历史统计 ============
train_sorted = train.sort_values('date_dt').reset_index(drop=True)

# 多级别group stats
for gcols, suffix in [
    (['meal_enc','area_enc','day_of_week'], 'maw'),
    (['meal_enc','area_enc'], 'ma'),
    (['meal_enc','day_of_week'], 'mw'),
]:
    stats = train_sorted.groupby(gcols)['volume'].agg(['mean','median','std']).reset_index()
    stats.columns = gcols + [f'{suffix}_mean', f'{suffix}_median', f'{suffix}_std']
    train = train.merge(stats, on=gcols, how='left')
    test = test.merge(stats, on=gcols, how='left')

# 最近N周统计
for weeks in [2, 3, 4, 6, 8]:
    cutoff = train_sorted['date_dt'].max() - pd.Timedelta(weeks=weeks)
    recent = train_sorted[train_sorted['date_dt'] >= cutoff]
    rs = recent.groupby(['meal_enc','area_enc','day_of_week'])['volume'].agg(['mean','std']).reset_index()
    rs.columns = ['meal_enc','area_enc','day_of_week', f'r{weeks}w_mean', f'r{weeks}w_std']
    train = train.merge(rs, on=['meal_enc','area_enc','day_of_week'], how='left')
    test = test.merge(rs, on=['meal_enc','area_enc','day_of_week'], how='left')

# demand_index 偏差
di_stats = train_sorted.groupby(['meal_enc','day_of_week'])['demand_index'].agg(['mean','std']).reset_index()
di_stats.columns = ['meal_enc','day_of_week','di_hist_mean','di_hist_std']
train = train.merge(di_stats, on=['meal_enc','day_of_week'], how='left')
test = test.merge(di_stats, on=['meal_enc','day_of_week'], how='left')
train['di_dev'] = (train['demand_index'] - train['di_hist_mean']) / train['di_hist_std'].clip(lower=1)
test['di_dev'] = (test['demand_index'] - test['di_hist_mean']) / test['di_hist_std'].clip(lower=1)

# ============ 特征列 ============
feature_cols = [
    'meal_enc', 'area_enc', 'day_of_week', 'day_of_month', 'month',
    'week_of_year', 'day_of_year',
    'weekday', 'is_weekend', 'semester_week', 'is_holiday', 'is_exam_week',
    'is_makeup_day', 'campus_event_level',
    'demand_index', 'temperature', 'feels_like', 'humidity', 'wind_speed',
    'rain_level', 'weather_enc',
    'menu_type_enc', 'menu_popularity', 'is_promotion',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'meal_area', 'weekday_meal', 'weekday_area', 'weekday_meal_area', 'is_workday',
    'di_x_pop', 'di_x_promo', 'di_x_area', 'di_x_meal',
    'semester_progress', 'exam_meal', 'holiday_meal', 'di_dev', 'di_hist_mean',
]
stat_cols = [c for c in train.columns if any(c.startswith(p) for p in ['maw_','ma_','mw_','r2w_','r3w_','r4w_','r6w_','r8w_'])]
feature_cols += stat_cols

X_train = train[feature_cols].values.astype(np.float32)
y_train = train['volume'].values
X_test = test[feature_cols].values.astype(np.float32)

# 验证集 (最后90天)
train['date_dt'] = pd.to_datetime(train['date'])
val_start = train['date_dt'].max() - pd.Timedelta(days=89)
val_mask = train['date_dt'] >= val_start
train_mask = ~val_mask
X_tr, y_tr = X_train[train_mask], y_train[train_mask]
X_val, y_val = X_train[val_mask], y_train[val_mask]

print(f"Features: {len(feature_cols)}, Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# ============ 多模型训练 ============
configs = [
    {'objective': 'regression_l1', 'learning_rate': 0.03, 'num_leaves': 127, 'min_child_samples': 15,
     'feature_fraction': 0.75, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.05, 'reg_lambda': 0.1, 'seed': 42},
    {'objective': 'regression', 'learning_rate': 0.03, 'num_leaves': 127, 'min_child_samples': 15,
     'feature_fraction': 0.75, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.05, 'reg_lambda': 0.1, 'seed': 42},
    {'objective': 'huber', 'learning_rate': 0.03, 'num_leaves': 95, 'min_child_samples': 20,
     'feature_fraction': 0.7, 'bagging_fraction': 0.85, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 0.2, 'seed': 123},
    {'objective': 'regression_l1', 'learning_rate': 0.02, 'num_leaves': 63, 'min_child_samples': 10,
     'feature_fraction': 0.85, 'bagging_fraction': 0.75, 'bagging_freq': 3, 'reg_alpha': 0.01, 'reg_lambda': 0.05, 'seed': 777},
]

val_preds = []
test_preds = []

for i, params in enumerate(configs):
    params.update({'metric': 'mae', 'verbose': -1, 'n_jobs': -1})
    
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=3000,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    
    vp = np.clip(model.predict(X_val), 0, None)
    val_preds.append(vp)
    
    # 用log空间再训一个
    dtrain_log = lgb.Dataset(X_tr, label=np.log1p(y_tr))
    dval_log = lgb.Dataset(X_val, label=np.log1p(y_val), reference=dtrain_log)
    model_log = lgb.train(params, dtrain_log, num_boost_round=3000,
                          valid_sets=[dval_log], callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    vp_log = np.clip(np.expm1(model_log.predict(X_val)), 0, None)
    val_preds.append(vp_log)
    
    # Full retrain
    dtrain_full = lgb.Dataset(X_train, label=y_train)
    model_full = lgb.train(params, dtrain_full, num_boost_round=model.best_iteration)
    test_preds.append(np.clip(model_full.predict(X_test), 0, None))
    
    dtrain_full_log = lgb.Dataset(X_train, label=np.log1p(y_train))
    model_full_log = lgb.train(params, dtrain_full_log, num_boost_round=model_log.best_iteration)
    test_preds.append(np.clip(np.expm1(model_full_log.predict(X_test)), 0, None))
    
    mae_i = np.mean(np.abs(y_val - vp))
    mae_log = np.mean(np.abs(y_val - vp_log))
    print(f"Model {i}: iter={model.best_iteration} MAE={mae_i:.2f} | log: iter={model_log.best_iteration} MAE={mae_log:.2f}")

# ============ 快速网格搜索最优融合 ============
def calc_score(pred, y_true):
    mae = np.mean(np.abs(y_true - pred))
    rmse = np.sqrt(np.mean((y_true - pred) ** 2))
    mape = np.mean(np.abs((y_true - pred) / np.clip(y_true, 1, None)))
    return (200 * max(0, 1 - mae/1000) + 150 * max(0, 1 - rmse/1500) + 150 * max(0, 1 - mape/0.5),
            mae, rmse, mape)

# 先试简单平均
avg_pred = np.mean(val_preds, axis=0)
s, mae, rmse, mape = calc_score(avg_pred, y_val)
print(f"\nSimple average: score={s:.1f}, MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.4f}")

# 贪心权重搜索
n_models = len(val_preds)
best_score = s
best_weights = np.ones(n_models) / n_models

# 逐步优化: 尝试增减每个模型的权重
for iteration in range(50):
    improved = False
    for i in range(n_models):
        for delta in [0.05, -0.05, 0.1, -0.1, 0.02, -0.02]:
            w = best_weights.copy()
            w[i] += delta
            if w[i] < 0:
                continue
            w = w / w.sum()
            pred = sum(w[j] * val_preds[j] for j in range(n_models))
            score, _, _, _ = calc_score(pred, y_val)
            if score > best_score:
                best_score = score
                best_weights = w
                improved = True
    if not improved:
        break

print(f"Optimized weights: {best_weights.round(4)}")
print(f"Best val score: {best_score:.2f}")

# 最终预测
val_final = sum(best_weights[j] * val_preds[j] for j in range(n_models))
s, mae, rmse, mape = calc_score(val_final, y_val)
mae_score = 200 * max(0, 1 - mae/1000)
rmse_score = 150 * max(0, 1 - rmse/1500)
mape_score = 150 * max(0, 1 - mape/0.5)
print(f"\n=== Final Validation ===")
print(f"MAE:  {mae:.2f} -> {mae_score:.1f}/200")
print(f"RMSE: {rmse:.2f} -> {rmse_score:.1f}/150")
print(f"MAPE: {mape:.4f} -> {mape_score:.1f}/150")
print(f"Total: {s:.1f}/500")

# 生成提交
test_final = sum(best_weights[j] * test_preds[j] for j in range(n_models))
test_final = np.clip(test_final, 0, None)

submission = test[['date', 'meal', 'canteen_area']].copy()
submission['volume'] = np.round(test_final, 2)
submission.to_csv('results.csv', index=False)

print(f"\nSubmission: {submission.shape}, range=[{submission['volume'].min():.1f}, {submission['volume'].max():.1f}]")
