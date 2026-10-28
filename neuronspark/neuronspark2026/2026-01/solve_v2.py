"""
NS-2026-01 智慧饭堂：人流与备餐预测 - 增强版
多策略融合: LightGBM多目标 + 精细特征工程 + 历史lag特征
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============ 1. 加载数据 ============
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
calendar = pd.read_csv('calendar.csv')
weather = pd.read_csv('weather.csv')
menu = pd.read_csv('menu.csv')

# ============ 2. 合并特征 ============
def merge_features(df):
    df = df.merge(calendar, on='date', how='left')
    df = df.merge(weather, on=['date', 'meal'], how='left')
    df = df.merge(menu, on=['date', 'meal', 'canteen_area'], how='left')
    return df

train = merge_features(train)
test = merge_features(test)

# ============ 3. 完整特征工程 ============
def build_features(df, train_df=None):
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'])
    
    # 基础时间特征
    df['day_of_week'] = df['date_dt'].dt.dayofweek
    df['day_of_month'] = df['date_dt'].dt.day
    df['month'] = df['date_dt'].dt.month
    df['day_of_year'] = df['date_dt'].dt.dayofyear
    df['week_of_year'] = df['date_dt'].dt.isocalendar().week.astype(int)
    
    # 编码
    meal_map = {'breakfast': 0, 'lunch': 1, 'dinner': 2}
    df['meal_enc'] = df['meal'].map(meal_map)
    area_map = {f'A0{i}': i for i in range(1, 7)}
    df['area_enc'] = df['canteen_area'].map(area_map)
    weather_map = {'clear': 0, 'cloudy': 1, 'rain': 2}
    df['weather_enc'] = df['weather'].map(weather_map)
    menu_type_map = {'noodle': 0, 'porridge': 1, 'light': 2, 'spicy': 3, 
                     'special': 4, 'rice_set': 5, 'western': 6}
    df['menu_type_enc'] = df['menu_type'].map(menu_type_map)
    
    # 周期性特征
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # 交互特征
    df['meal_area'] = df['meal_enc'] * 10 + df['area_enc']
    df['weekday_meal'] = df['day_of_week'] * 3 + df['meal_enc']
    df['weekday_area'] = df['day_of_week'] * 6 + df['area_enc']
    df['weekday_meal_area'] = df['day_of_week'] * 18 + df['meal_enc'] * 6 + df['area_enc']
    
    # 工作日
    df['is_workday'] = ((df['is_weekend'] == 0) & (df['is_holiday'] == 0)).astype(int)
    
    # demand_index 交互
    df['di_x_pop'] = df['demand_index'] * df['menu_popularity']
    df['di_x_promo'] = df['demand_index'] * df['is_promotion']
    df['di_x_area'] = df['demand_index'] * df['area_enc']
    df['di_x_meal'] = df['demand_index'] * df['meal_enc']
    
    # 天气交互
    df['temp_x_rain'] = df['temperature'] * df['rain_level']
    df['humidity_x_rain'] = df['humidity'] * df['rain_level']
    
    # 学期进度 (semester_week / max)
    df['semester_progress'] = df['semester_week'] / 22.0
    
    # 考试周+饭点交互 (考试周时早餐可能减少)
    df['exam_meal'] = df['is_exam_week'] * df['meal_enc']
    df['holiday_meal'] = df['is_holiday'] * df['meal_enc']
    
    return df

train = build_features(train)
test = build_features(test)

# ============ 4. 历史统计特征 (lag features) ============
train_sorted = train.sort_values('date_dt').reset_index(drop=True)

# 构造历史统计: 对每个 (meal, area, weekday) 组
def compute_group_stats(source_df, group_cols, target_col='volume'):
    stats = source_df.groupby(group_cols)[target_col].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
    stats.columns = group_cols + [f'grp_{"_".join(group_cols[1:])}_mean', 
                                   f'grp_{"_".join(group_cols[1:])}_median',
                                   f'grp_{"_".join(group_cols[1:])}_std',
                                   f'grp_{"_".join(group_cols[1:])}_min',
                                   f'grp_{"_".join(group_cols[1:])}_max']
    return stats

# 多级别统计
stats1 = compute_group_stats(train_sorted, ['meal_enc', 'area_enc', 'day_of_week'])
stats2 = compute_group_stats(train_sorted, ['meal_enc', 'area_enc'])
stats3 = compute_group_stats(train_sorted, ['meal_enc', 'day_of_week'])

for s in [stats1, stats2, stats3]:
    train = train.merge(s, on=s.columns[:len(s.columns)-5].tolist(), how='left')
    test = test.merge(s, on=s.columns[:len(s.columns)-5].tolist(), how='left')

# 最近N周统计
for weeks in [2, 3, 4, 6, 8]:
    cutoff = train_sorted['date_dt'].max() - pd.Timedelta(weeks=weeks)
    recent = train_sorted[train_sorted['date_dt'] >= cutoff]
    rs = recent.groupby(['meal_enc', 'area_enc', 'day_of_week'])['volume'].agg(['mean', 'std']).reset_index()
    rs.columns = ['meal_enc', 'area_enc', 'day_of_week', f'recent_{weeks}w_mean', f'recent_{weeks}w_std']
    train = train.merge(rs, on=['meal_enc', 'area_enc', 'day_of_week'], how='left')
    test = test.merge(rs, on=['meal_enc', 'area_enc', 'day_of_week'], how='left')

# 最近N周按 (meal, area) 的均值 (不分weekday)
for weeks in [2, 4, 8]:
    cutoff = train_sorted['date_dt'].max() - pd.Timedelta(weeks=weeks)
    recent = train_sorted[train_sorted['date_dt'] >= cutoff]
    rs = recent.groupby(['meal_enc', 'area_enc'])['volume'].mean().reset_index()
    rs.columns = ['meal_enc', 'area_enc', f'recent_{weeks}w_ma_mean']
    train = train.merge(rs, on=['meal_enc', 'area_enc'], how='left')
    test = test.merge(rs, on=['meal_enc', 'area_enc'], how='left')

# demand_index 的历史统计 per (meal, weekday)
di_stats = train_sorted.groupby(['meal_enc', 'day_of_week'])['demand_index'].agg(['mean','std']).reset_index()
di_stats.columns = ['meal_enc', 'day_of_week', 'di_hist_mean', 'di_hist_std']
train = train.merge(di_stats, on=['meal_enc', 'day_of_week'], how='left')
test = test.merge(di_stats, on=['meal_enc', 'day_of_week'], how='left')

# demand_index 相对偏差
train['di_deviation'] = (train['demand_index'] - train['di_hist_mean']) / train['di_hist_std'].clip(lower=1)
test['di_deviation'] = (test['demand_index'] - test['di_hist_mean']) / test['di_hist_std'].clip(lower=1)

# ============ 5. 定义特征列 ============
feature_cols = [
    # 基础编码
    'meal_enc', 'area_enc', 'day_of_week', 'day_of_month', 'month',
    'week_of_year', 'day_of_year',
    # calendar
    'weekday', 'is_weekend', 'semester_week', 'is_holiday', 'is_exam_week',
    'is_makeup_day', 'campus_event_level',
    # weather
    'demand_index', 'temperature', 'feels_like', 'humidity', 'wind_speed',
    'rain_level', 'weather_enc',
    # menu
    'menu_type_enc', 'menu_popularity', 'is_promotion',
    # 周期性
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
    # 交互
    'meal_area', 'weekday_meal', 'weekday_area', 'weekday_meal_area', 'is_workday',
    'di_x_pop', 'di_x_promo', 'di_x_area', 'di_x_meal',
    'temp_x_rain', 'humidity_x_rain',
    'semester_progress', 'exam_meal', 'holiday_meal',
    # demand_index 偏差
    'di_deviation', 'di_hist_mean',
]

# 添加所有统计特征
stat_cols = [c for c in train.columns if c.startswith('grp_') or c.startswith('recent_')]
feature_cols += stat_cols

# 确保所有特征都存在
for col in feature_cols:
    if col not in train.columns:
        print(f"WARNING: {col} not in train")
        feature_cols.remove(col)
    if col not in test.columns:
        print(f"WARNING: {col} not in test")
        feature_cols.remove(col)

X_train = train[feature_cols].values.astype(np.float32)
y_train = train['volume'].values
X_test = test[feature_cols].values.astype(np.float32)

print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
print(f"Test samples: {X_test.shape[0]}")

# ============ 6. 时间序列验证 ============
train['date_dt'] = pd.to_datetime(train['date'])
val_start = train['date_dt'].max() - pd.Timedelta(days=89)
val_mask = train['date_dt'] >= val_start
train_mask = ~val_mask

X_tr = X_train[train_mask]
y_tr = y_train[train_mask]
X_val = X_train[val_mask]
y_val = y_train[val_mask]

print(f"Validation split: train={X_tr.shape[0]}, val={X_val.shape[0]}")

# ============ 7. 多配置 LightGBM 训练 ============
configs = [
    {  # MAE优化
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
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
    },
    {  # RMSE优化
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
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
    },
    {  # Huber loss (robust)
        'objective': 'huber',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 95,
        'min_child_samples': 20,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.2,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 123,
    },
    {  # MAE with different structure
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 63,
        'min_child_samples': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.75,
        'bagging_freq': 3,
        'reg_alpha': 0.01,
        'reg_lambda': 0.05,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 777,
    },
    {  # MAPE优化 (用 log transform)
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 95,
        'min_child_samples': 15,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.05,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 314,
    },
]

models_val = []
val_preds = []
test_preds = []

for i, params in enumerate(configs):
    use_log = (i == 4)  # 最后一个用 log transform
    
    y_tr_i = np.log1p(y_tr) if use_log else y_tr
    y_val_i = np.log1p(y_val) if use_log else y_val
    y_train_i = np.log1p(y_train) if use_log else y_train
    
    dtrain = lgb.Dataset(X_tr, label=y_tr_i)
    dval = lgb.Dataset(X_val, label=y_val_i, reference=dtrain)
    
    model = lgb.train(
        params, dtrain, num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    vp = model.predict(X_val)
    if use_log:
        vp = np.expm1(vp)
    vp = np.clip(vp, 0, None)
    val_preds.append(vp)
    
    # Retrain on full data
    dtrain_full = lgb.Dataset(X_train, label=y_train_i)
    model_full = lgb.train(params, dtrain_full, num_boost_round=model.best_iteration)
    
    tp = model_full.predict(X_test)
    if use_log:
        tp = np.expm1(tp)
    tp = np.clip(tp, 0, None)
    test_preds.append(tp)
    
    mae_i = np.mean(np.abs(y_val - vp))
    print(f"Model {i} ({params['objective']}, log={use_log}): best_iter={model.best_iteration}, val MAE={mae_i:.2f}")

# ============ 8. 最优融合权重搜索 ============
from scipy.optimize import minimize

def score_fn(weights, preds, y_true):
    w = np.array(weights)
    w = w / w.sum()  # normalize
    combined = sum(w[i] * preds[i] for i in range(len(preds)))
    combined = np.clip(combined, 0, None)
    mae = np.mean(np.abs(y_true - combined))
    rmse = np.sqrt(np.mean((y_true - combined) ** 2))
    mape = np.mean(np.abs((y_true - combined) / np.clip(y_true, 1, None)))
    score = 200 * max(0, 1 - mae/1000) + 150 * max(0, 1 - rmse/1500) + 150 * max(0, 1 - mape/0.5)
    return -score  # minimize negative score

from scipy.optimize import differential_evolution

bounds = [(0.01, 1.0)] * len(val_preds)
result = differential_evolution(score_fn, bounds, args=(val_preds, y_val), 
                                seed=42, maxiter=1000, tol=1e-8)
opt_weights = np.array(result.x)
opt_weights = opt_weights / opt_weights.sum()
print(f"\nOptimal weights: {opt_weights.round(4)}")
print(f"Optimal val score: {-result.fun:.2f}")

# ============ 9. 生成融合预测 ============
val_combined = sum(opt_weights[i] * val_preds[i] for i in range(len(val_preds)))
val_combined = np.clip(val_combined, 0, None)

mae = np.mean(np.abs(y_val - val_combined))
rmse = np.sqrt(np.mean((y_val - val_combined) ** 2))
mape = np.mean(np.abs((y_val - val_combined) / np.clip(y_val, 1, None)))

mae_score = 200 * max(0, 1 - mae / 1000.0)
rmse_score = 150 * max(0, 1 - rmse / 1500.0)
mape_score = 150 * max(0, 1 - mape / 0.50)
total_score = mae_score + rmse_score + mape_score

print(f"\n=== Ensemble Validation Results ===")
print(f"MAE:  {mae:.2f} -> score: {mae_score:.1f}/200")
print(f"RMSE: {rmse:.2f} -> score: {rmse_score:.1f}/150")
print(f"MAPE: {mape:.4f} -> score: {mape_score:.1f}/150")
print(f"Total: {total_score:.1f}/500")

# Final test prediction
test_combined = sum(opt_weights[i] * test_preds[i] for i in range(len(test_preds)))
test_combined = np.clip(test_combined, 0, None)

# ============ 10. 提交 ============
submission = test[['date', 'meal', 'canteen_area']].copy()
submission['volume'] = np.round(test_combined, 2)
submission.to_csv('results.csv', index=False)

print(f"\n=== Submission Generated ===")
print(f"Shape: {submission.shape}")
print(f"Volume range: [{submission['volume'].min():.2f}, {submission['volume'].max():.2f}]")
print(f"Volume mean: {submission['volume'].mean():.2f}")
