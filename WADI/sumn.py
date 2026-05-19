import pandas as pd
import numpy as np
import os
import json
import argparse  # 新增：用于解析命令行比例参数
from collections import defaultdict
import warnings
# 1. [新增] 引入新的评估指标库
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

# 忽略警告
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 全局配置区域
# ==============================================================================
# 默认配置值（通过命令行参数覆盖）
DEFAULT_CONFIG = {
    'data_file': r"WADI_attackdataLABLE.csv",
    'rules_file': "merged_tank_rules.csv",
    'bounds_file': "tank_bounds.json",
    'constant_file': "constant_devices.json",
    'std_json_file': "sensor_min_std_filtered.json",
    'output_file': "WADI_Final_Fusion_Result.csv",

    # [引擎 A: 模型]
    'min_anomaly_length_model': 60,
    'default_threshold': (3.0, 3.0),

    # [引擎 B: 规则]
    'trend_window': 40,
    'fluctuation_val': 0.5,
    'search_buffer': 60,
    'threshold_pct': 0.05,
    'max_gap_size_rule': 10,
    'std_check_window': 120
}

# 数据文件
DATA_FILE = DEFAULT_CONFIG['data_file']

# 规则配置文件
RULES_FILE = DEFAULT_CONFIG['rules_file']
BOUNDS_FILE = DEFAULT_CONFIG['bounds_file']
CONSTANT_FILE = DEFAULT_CONFIG['constant_file']
STD_JSON_FILE = DEFAULT_CONFIG['std_json_file']

# 输出文件
OUTPUT_FILE = DEFAULT_CONFIG['output_file']

# --- 参数配置 ---
# [引擎 A: 模型]
MIN_ANOMALY_LENGTH_MODEL = DEFAULT_CONFIG['min_anomaly_length_model']
DEFAULT_THRESHOLD = DEFAULT_CONFIG['default_threshold']

# [引擎 B: 规则]
TREND_WINDOW = DEFAULT_CONFIG['trend_window']
FLUCTUATION_VAL = DEFAULT_CONFIG['fluctuation_val']
SEARCH_BUFFER = DEFAULT_CONFIG['search_buffer']
THRESHOLD_PCT = DEFAULT_CONFIG['threshold_pct']
MAX_GAP_SIZE_RULE = DEFAULT_CONFIG['max_gap_size_rule']
STD_CHECK_WINDOW = DEFAULT_CONFIG['std_check_window']

# 设备字典
DEVICE_TYPE = {
    '1_AIT_001_PV': 'Sensor', '1_AIT_002_PV': 'Sensor', '1_AIT_003_PV': 'Sensor', '1_AIT_004_PV': 'Sensor',
    '1_AIT_005_PV': 'Sensor', '1_FIT_001_PV': 'Sensor', '1_LS_001_AL': 'Actuator', '1_LS_002_AL': 'Actuator',
    '1_LT_001_PV': 'Sensor', '1_MV_001_STATUS': 'Actuator', '1_MV_002_STATUS': 'Actuator', '1_MV_003_STATUS': 'Actuator',
    '1_MV_004_STATUS': 'Actuator', '1_P_001_STATUS': 'Actuator', '1_P_002_STATUS': 'Actuator', '1_P_003_STATUS': 'Actuator',
    '1_P_004_STATUS': 'Actuator', '1_P_005_STATUS': 'Actuator', '1_P_006_STATUS': 'Actuator', '2_DPIT_001_PV': 'Sensor',
    '2_FIC_101_CO': 'Sensor', '2_FIC_101_PV': 'Sensor', '2_FIC_101_SP': 'Sensor', '2_FIC_201_CO': 'Sensor',
    '2_FIC_201_PV': 'Sensor', '2_FIC_201_SP': 'Sensor', '2_FIC_301_CO': 'Sensor', '2_FIC_301_PV': 'Sensor',
    '2_FIC_301_SP': 'Sensor', '2_FIC_401_CO': 'Sensor', '2_FIC_401_PV': 'Sensor', '2_FIC_401_SP': 'Sensor',
    '2_FIC_501_CO': 'Sensor', '2_FIC_501_PV': 'Sensor', '2_FIC_501_SP': 'Sensor', '2_FIC_601_CO': 'Sensor',
    '2_FIC_601_PV': 'Sensor', '2_FIC_601_SP': 'Sensor', '2_FIT_001_PV': 'Sensor', '2_FIT_002_PV': 'Sensor',
    '2_FIT_003_PV': 'Sensor', '2_FQ_101_PV': 'Sensor', '2_FQ_201_PV': 'Sensor', '2_FQ_301_PV': 'Sensor',
    '2_FQ_401_PV': 'Sensor', '2_FQ_501_PV': 'Sensor', '2_FQ_601_PV': 'Sensor', '2_LS_001_AL': 'Sensor',
    '2_LS_002_AL': 'Sensor', '2_LS_101_AH': 'Actuator', '2_LS_101_AL': 'Actuator', '2_LS_201_AH': 'Actuator',
    '2_LS_201_AL': 'Actuator', '2_LS_301_AH': 'Actuator', '2_LS_301_AL': 'Actuator', '2_LS_401_AH': 'Actuator',
    '2_LS_401_AL': 'Actuator', '2_LS_501_AH': 'Actuator', '2_LS_501_AL': 'Actuator', '2_LS_601_AH': 'Actuator',
    '2_LS_601_AL': 'Actuator', '2_LT_001_PV': 'Sensor', '2_LT_002_PV': 'Sensor', '2_MCV_007_CO': 'Sensor',
    '2_MCV_101_CO': 'Sensor', '2_MCV_201_CO': 'Sensor', '2_MCV_301_CO': 'Sensor', '2_MCV_401_CO': 'Sensor',
    '2_MCV_501_CO': 'Sensor', '2_MCV_601_CO': 'Sensor', '2_MV_001_STATUS': 'Actuator', '2_MV_002_STATUS': 'Actuator',
    '2_MV_003_STATUS': 'Actuator', '2_MV_004_STATUS': 'Actuator', '2_MV_005_STATUS': 'Actuator',
    '2_MV_006_STATUS': 'Actuator',
    '2_MV_009_STATUS': 'Actuator', '2_MV_101_STATUS': 'Actuator', '2_MV_201_STATUS': 'Actuator',
    '2_MV_301_STATUS': 'Actuator',
    '2_MV_401_STATUS': 'Actuator', '2_MV_501_STATUS': 'Actuator', '2_MV_601_STATUS': 'Actuator',
    '2_P_001_STATUS': 'Sensor',
    '2_P_002_STATUS': 'Sensor', '2_P_003_SPEED': 'Sensor', '2_P_003_STATUS': 'Actuator', '2_P_004_SPEED': 'Sensor',
    '2_P_004_STATUS': 'Actuator', '2_PIC_003_CO': 'Sensor', '2_PIC_003_PV': 'Sensor', '2_PIC_003_SP': 'Actuator',
    '2_PIT_001_PV': 'Sensor', '2_PIT_002_PV': 'Sensor', '2_PIT_003_PV': 'Sensor', '2_SV_101_STATUS': 'Actuator',
    '2_SV_201_STATUS': 'Actuator', '2_SV_301_STATUS': 'Actuator', '2_SV_401_STATUS': 'Actuator',
    '2_SV_501_STATUS': 'Actuator',
    '2_SV_601_STATUS': 'Actuator', '2A_AIT_001_PV': 'Sensor', '2A_AIT_002_PV': 'Sensor', '2A_AIT_003_PV': 'Sensor',
    '2A_AIT_004_PV': 'Sensor', '2B_AIT_001_PV': 'Sensor', '2B_AIT_002_PV': 'Sensor', '2B_AIT_003_PV': 'Sensor',
    '2B_AIT_004_PV': 'Sensor', '3_AIT_001_PV': 'Actuator', '3_AIT_002_PV': 'Actuator', '3_AIT_003_PV': 'Sensor',
    '3_AIT_004_PV': 'Sensor', '3_AIT_005_PV': 'Sensor', '3_FIT_001_PV': 'Sensor', '3_LS_001_AL': 'Actuator',
    '3_LT_001_PV': 'Sensor', '3_MV_001_STATUS': 'Actuator', '3_MV_002_STATUS': 'Actuator',
    '3_MV_003_STATUS': 'Actuator',
    '3_P_001_STATUS': 'Actuator', '3_P_002_STATUS': 'Actuator', '3_P_003_STATUS': 'Actuator',
    '3_P_004_STATUS': 'Actuator',
    'LEAK_DIFF_PRESSURE': 'Sensor', 'PLANT_START_STOP_LOG': 'Actuator', 'TOTAL_CONS_REQUIRED_FLOW': 'Sensor',
}

# [引擎 A] 模型参数
MODEL_THRESHOLDS = {
    '1_FIT_001_PV': (50.0, 50.0), '3_AIT_003_PV': (10.0, 10.0), '2_DPIT_001_PV': (20.0, 5),
    '3_AIT_002_PV': (2.5, 2.5), '2_PIT_001_PV': (2.5, 2.5), '2_FIC_201_PV': (15.0, 15.0),
    '2_FIC_601_PV': (6.0, 6.0), '2_FIC_201_CO': (5.5, 5.5), '2B_AIT_003_PV': (6.0, 6.0),
    '2_FIC_301_CO': (5.5, 5.5), '2_FIC_501_PV': (5.0, 5.0), '2_FIC_101_PV': (10.0, 10.0),
    '2A_AIT_004_PV': (5.0, 7.0), '2_FIC_301_PV': (15.0, 15.0), '2_FIC_401_PV': (5.0, 5.0),
    '2_FIC_101_SP': (3.0, 3.0), '2A_AIT_001_PV': (3.0, 6.0), '2A_AIT_003_PV': (3.0, 6.0),
    '2_FIC_301_CO': (3.0, 3.0), '2_FIC_401_SP': (3.0, 3.0), '2_MV_501_STATUS': (3.0, 3.0),
    '2_MV_601_STATUS': (3.0, 3.0), '2_FIC_301_SP': (3.0, 3.0), '2_FIC_501_SP': (3.0, 3.0),
    '2_MV_201_STATUS': (3.0, 3.0), '2_MV_301_STATUS': (3.0, 3.0), '2B_AIT_004_PV': (3.0, 6.0),
    '1_P_005_STATUS': (4.0, 4.0), '2_FIT_003_PV': (4.0, 4.0),
}

MODELS = {
    '2_FIT_003_PV': {'type': 'Type-L', 'intercept': 0.06227, 'mean': 0.0, 'std': 0.078145, 'coeffs': {'2_DPIT_001_PV': -0.00008, '2_MV_006_STATUS': 0.01926, '2_P_003_SPEED': -0.00653, '2_P_003_STATUS': 0.13825, '2_P_004_SPEED': 0.36622, '2_PIC_003_CO': -0.00021, '2_PIC_003_PV': -0.34076, '2_PIT_002_PV': -0.00024, '2_PIT_003_PV': 0.31442}},
    '2_FIC_101_SP': {'type': 'Type-L', 'intercept': 0.00001, 'mean': 0.0, 'std': 0.001939, 'coeffs': {'2_FIC_201_SP': -0.00001, '2_FIC_301_SP': -0.00012, '2_FIC_401_SP': 0.00012, '2_FIC_601_SP': -0.00005}},
    '2A_AIT_001_PV': {'type': 'Type-L', 'intercept': -0.00001, 'mean': 0.0, 'std': 0.000192, 'coeffs': {'2A_AIT_003_PV': 0.00001, '2B_AIT_001_PV': 0.00001, '2B_AIT_003_PV': 0.0}},
    '2A_AIT_003_PV': {'type': 'Type-L', 'intercept': -0.00756, 'mean': 0.0, 'std': 0.003666, 'coeffs': {'2A_AIT_001_PV': 0.00148, '2A_AIT_004_PV': 0.00001, '2B_AIT_001_PV': -0.00083, '2B_AIT_003_PV': 0.00038}},
    '2B_AIT_003_PV': {'type': 'Type-L', 'intercept': -0.00189, 'mean': 0.0, 'std': 0.008957, 'coeffs': {'2A_AIT_001_PV': -0.00005, '2A_AIT_003_PV': 0.0001, '2B_AIT_004_PV': 0.0}},
    '1_FIT_001_PV': {'type': 'Type-C', 'intercept': 3790797906.83, 'mean': 0.0, 'std': 0.029178, 'coeffs': {'1_MV_001_STATUS': -0.6432, '1_MV_001_STATUS^2': 0.371, 'd(1_MV_001_STATUS)': -0.0224, '1_P_001_STATUS': 7370014.6, '1_P_001_STATUS^2': 1323345683.4, 'd(1_P_001_STATUS)': 3397937.7, '1_P_003_STATUS': -5693566875.1, '1_P_003_STATUS^2': 572053270.5, 'd(1_P_003_STATUS)': -3397937.7}},
    '2_DPIT_001_PV': {'type': 'Type-C', 'intercept': -14618.4395, 'mean': 0.0, 'std': 18.360441, 'coeffs': {'2_FIT_003_PV': -20.9679, '2_FIT_003_PV^2': -1.2701, 'd(2_FIT_003_PV)': -0.8704, '2_PIT_001_PV': 200.4009, '2_PIT_001_PV^2': -0.6423, 'd(2_PIT_001_PV)': -4.8674, '2_PIT_002_PV': 21.1866, '2_PIT_002_PV^2': -0.0469, 'd(2_PIT_002_PV)': 1.1254}},
    '2_FIC_201_CO': {'type': 'Type-L', 'intercept': -0.04324, 'mean': 0.0, 'std': 0.327146, 'coeffs': {'2_FIC_501_CO': -0.00057, '2_MV_201_STATUS': 0.02486, '2_MV_501_STATUS': 0.02699}},
    '2_FIC_301_CO': {'type': 'Type-L', 'intercept': -0.02601, 'mean': 0.0, 'std': 0.342377, 'coeffs': {'2_FIC_601_CO': -0.0003, '2_MV_301_STATUS': 0.02248, '2_MV_601_STATUS': 0.00663}},
    '2_FIC_401_CO': {'type': 'Type-L', 'intercept': -0.04343, 'mean': 0.0, 'std': 0.435819, 'coeffs': {'2_FIC_101_CO': -0.00081, '2_MV_101_STATUS': 0.03545, '2_MV_401_STATUS': 0.03019}},
    '2_FIC_401_SP': {'type': 'Type-L', 'intercept': 0.00001, 'mean': 0.0, 'std': 0.001697, 'coeffs': {'2_FIC_101_SP': 0.0, '2_FIC_201_SP': -0.00001, '2_FIC_501_SP': -0.00002}},
    '2_MV_501_STATUS': {'type': 'Type-L', 'intercept': 0.00011, 'mean': 0.0, 'std': 0.034998, 'coeffs': {'2_FIC_201_CO': 0.0, '2_FIC_501_CO': 0.0, '2_MCV_201_CO': 0.0}},
    '2_MV_601_STATUS': {'type': 'Type-L', 'intercept': 0.00012, 'mean': 0.0, 'std': 0.022152, 'coeffs': {'2_FIC_301_CO': 0.0, '2_FIC_601_CO': 0.0, '2_MCV_101_CO': 0.0}},
    '2_PIT_001_PV': {'type': 'Type-L', 'intercept': 0.33793, 'mean': 0.0, 'std': 0.342682, 'coeffs': {'2_DPIT_001_PV': -0.00013, '2_LT_002_PV': 0.00265, '2_PIT_002_PV': -0.00234}},
    '1_P_005_STATUS': {'type': 'Type-C', 'intercept': 1.0006, 'mean': 0.0, 'std': 0.026009, 'coeffs': {'2_FIT_001_PV': 0.0288, '2_FIT_001_PV^2': 0.004, 'd(2_FIT_001_PV)': 0.0101, '2_MV_003_STATUS': -0.4584, '2_MV_003_STATUS^2': 0.4578, 'd(2_MV_003_STATUS)': -0.003}},
    '2_FIC_301_SP': {'type': 'Type-L', 'intercept': 0.0, 'mean': 0.0, 'std': 0.002006, 'coeffs': {'2_FIC_101_SP': 0.00001, '2_FIC_501_SP': -0.00003}},
    '2_FIC_501_SP': {'type': 'Type-L', 'intercept': 0.00001, 'mean': 0.0, 'std': 0.001951, 'coeffs': {'2_FIC_301_SP': -0.00004, '2_FIC_401_SP': 0.00001}},
    '2_MV_201_STATUS': {'type': 'Type-L', 'intercept': 0.00029, 'mean': 0.0, 'std': 0.02272, 'coeffs': {'2_FIC_201_CO': 0.0, '2_MCV_501_CO': 0.0}},
    '2_MV_301_STATUS': {'type': 'Type-L', 'intercept': 0.00024, 'mean': 0.0, 'std': 0.02145, 'coeffs': {'2_FIC_301_CO': 0.0, '2_MCV_401_CO': 0.0}},
    '2A_AIT_004_PV': {'type': 'Type-L', 'intercept': -0.46311, 'mean': 0.0, 'std': 0.13567, 'coeffs': {'2A_AIT_003_PV': 0.03658, '2B_AIT_004_PV': 0.00031}},
    '2B_AIT_004_PV': {'type': 'Type-L', 'intercept': -1.05467, 'mean': 0.0, 'std': 0.793353, 'coeffs': {'2A_AIT_004_PV': 0.00092, '2B_AIT_003_PV': 0.07135}},
    '2_FIC_101_PV': {'type': 'Type-C', 'intercept': -0.0001, 'mean': 0.0, 'std': 0.006454, 'coeffs': {'2_FQ_101_PV': 1.0015, '2_FQ_101_PV^2': -0.002, 'd(2_FQ_101_PV)': -0.0175}},
    '2_FIC_201_PV': {'type': 'Type-C', 'intercept': 0.0001, 'mean': 0.0, 'std': 0.004414, 'coeffs': {'2_FQ_201_PV': 0.9991, '2_FQ_201_PV^2': 0.0021, 'd(2_FQ_201_PV)': -0.015}},
    '2_FIC_301_PV': {'type': 'Type-C', 'intercept': 0.0, 'mean': 0.0, 'std': 0.004649, 'coeffs': {'2_FQ_301_PV': 1.0005, '2_FQ_301_PV^2': -0.0016, 'd(2_FQ_301_PV)': -0.0148}},
    '2_FIC_401_PV': {'type': 'Type-C', 'intercept': 0.0, 'mean': 0.0, 'std': 0.005884, 'coeffs': {'2_FQ_401_PV': 1.0001, '2_FQ_401_PV^2': 0.0007, 'd(2_FQ_401_PV)': -0.0098}},
    '2_FIC_501_PV': {'type': 'Type-C', 'intercept': 0.0, 'mean': 0.0, 'std': 0.004342, 'coeffs': {'2_FQ_501_PV': 1.0006, '2_FQ_501_PV^2': -0.0016, 'd(2_FQ_501_PV)': -0.0193}},
    '2_FIC_601_PV': {'type': 'Type-C', 'intercept': 0.0, 'mean': 0.0, 'std': 0.005132, 'coeffs': {'2_FQ_601_PV': 1.0007, '2_FQ_601_PV^2': -0.0015, 'd(2_FQ_601_PV)': -0.0192}},
    '3_AIT_002_PV': {'type': 'Type-L', 'intercept': -0.70941, 'mean': 0.0, 'std': 122.183951, 'coeffs': {'3_FIT_001_PV': 1.39349}},
    '3_AIT_003_PV': {'type': 'Type-C', 'intercept': 15.899, 'mean': 0.0, 'std': 3.95286, 'coeffs': {'3_AIT_004_PV': -0.0159, 'd(3_AIT_004_PV)': 0.0001}}
}


# ==============================================================================
# 1. 引擎 A: 模型检测函数
# ==============================================================================
def detect_anomalies_model(df, models, threshold_map):
    print("🔵 [Engine A] 运行回归模型检测...")
    required_cols = set()
    for target, m in models.items():
        if target in df.columns:
            required_cols.add(target)
            for k in m['coeffs'].keys():
                raw_k = k.replace("d(", "").replace(")", "").replace("^2", "")
                if raw_k in df.columns: required_cols.add(raw_k)

    df_subset = df[list(required_cols)]
    df_rolling = df_subset.rolling(window=10).mean().fillna(0)
    df_diff = df_subset.diff().fillna(0)

    df_status = pd.DataFrame(index=df.index)
    model_causes = pd.Series([""] * len(df), index=df.index, dtype=object)

    for target, config in models.items():
        if target not in df.columns: continue

        y_pred = pd.Series(config['intercept'], index=df.index)
        for feat_name, weight in config['coeffs'].items():
            raw_feat = feat_name.replace("d(", "").replace(")", "").replace("^2", "")
            if raw_feat not in df.columns: continue

            if config['type'] == 'Type-L':
                y_pred += weight * df_rolling[raw_feat]
            else:
                if feat_name.startswith("d("):
                    y_pred += weight * df_diff[raw_feat]
                elif feat_name.endswith("^2"):
                    y_pred += weight * (df[raw_feat] ** 2)
                else:
                    y_pred += weight * df[raw_feat]

        if config['type'] == 'Type-L':
            residual = df[target].diff() - y_pred
        else:
            residual = df[target] - y_pred

        lb_mult, ub_mult = threshold_map.get(target, DEFAULT_THRESHOLD)
        lower = config['mean'] - (lb_mult * config['std'])
        upper = config['mean'] + (ub_mult * config['std'])

        is_normal = (residual >= lower) & (residual <= upper)
        df_status[target] = np.where(is_normal, 1, -1)
        df_status[target].iloc[:10] = 1

    if df_status.empty:
        return np.ones(len(df)), model_causes

    y_raw = df_status.min(axis=1).values
    anom_indices = np.where(y_raw == -1)[0]
    if len(anom_indices) > 0:
        subset = df_status.iloc[anom_indices]
        mask = (subset == -1)
        causes = mask.dot("Model:" + mask.columns + "; ").str.rstrip("; ")
        model_causes.iloc[anom_indices] = causes

    return y_raw, model_causes


def filter_anomalies_by_length(y_pred, min_len):
    filtered = y_pred.copy()
    n = len(filtered)
    i = 0
    while i < n:
        if filtered[i] == -1:
            start = i
            while i < n and filtered[i] == -1: i += 1
            if (i - start) < min_len: filtered[start:i] = 1
        else:
            i += 1
    return filtered


# ==============================================================================
# 2. 引擎 B: 规则检测逻辑 (完全集成)
# ==============================================================================
def find_events_robust(df, device_type_dict):
    print("🔍 [Rule] 提取事件...")
    events = []
    actuator_cols = [col for col, dtype in device_type_dict.items() if dtype == 'Actuator']
    valid_cols = [c for c in actuator_cols if c in df.columns]
    df_subset = df[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    df_diff = df_subset.diff()
    changed_indices = df_diff.any(axis=1).to_numpy().nonzero()[0]

    for index in changed_indices:
        if index == 0: continue
        changed_cols_now = df_diff.iloc[index].dropna().index
        for col in changed_cols_now:
            prev = df_subset.iloc[index - 1][col]
            curr = df_subset.iloc[index][col]
            if prev == 1 and curr == 2:
                events.append({'idx': index, 'device': col, 'state': 2})
            elif prev == 2 and curr == 1:
                events.append({'idx': index, 'device': col, 'state': 1})
            elif curr == 0 and prev != 0:
                limit = min(index + 500, len(df_subset))
                future = df_subset[col].iloc[index + 1:limit].values
                nz = np.nonzero(future)[0]
                if len(nz) > 0:
                    final = df_subset.iloc[index + nz[0] + 1][col]
                    if prev == 1 and final == 2:
                        events.append({'idx': index, 'device': col, 'state': 2})
                    elif prev == 2 and final == 1:
                        events.append({'idx': index, 'device': col, 'state': 1})
    return events


def parse_rules_reversed(filepath):
    if not os.path.exists(filepath): return {}, {}
    df = pd.read_csv(filepath)
    rule_dict_s1 = defaultdict(list)
    action_map_s2 = defaultdict(list)
    for _, row in df.iterrows():
        ant = row['Antecedent (Tank Condition)']
        cons_str = row['Consequent (Merged Actions)']
        actions = []
        raw_actions = cons_str.split(', ')
        for item in raw_actions:
            clean_item = item.split(' (')[0]
            if "_Open" in clean_item:
                actions.append({'device': clean_item.replace("_Open", ""), 'target': 2})
            elif "_Close" in clean_item:
                actions.append({'device': clean_item.replace("_Close", ""), 'target': 1})
        rule_dict_s1[ant] = actions
        for act in actions:
            key = (act['device'], act['target'])
            action_map_s2[key].append(
                {'condition': ant, 'partners': [a for a in actions if a['device'] != act['device']]})
    return rule_dict_s1, action_map_s2


def get_next_change_index(df_subset, device, current_idx, total_rows):
    current_val = df_subset.at[current_idx, device]
    future = df_subset[device].iloc[current_idx + 1:].values
    diff_mask = future != current_val
    if diff_mask.any(): return current_idx + 1 + diff_mask.argmax()
    return total_rows - 1


def find_end_s1(df, start_idx, total_rows, missing_device, missing_target, tank_bool_series):
    curr = start_idx
    limit = min(start_idx + 86400, total_rows)
    while curr < limit:
        if df.at[curr, missing_device] == missing_target: return curr
        if not tank_bool_series[curr]: return curr
        curr += 1
    return limit


def find_end_s3(df, start_idx, total_rows, anchor_dev, anchor_orig_state, missing_dev, missing_target):
    curr = start_idx
    limit = min(start_idx + 86400, total_rows)
    while curr < limit:
        if df.at[curr, missing_dev] == missing_target: return curr
        if df.at[curr, anchor_dev] != anchor_orig_state: return curr
        curr += 1
    return limit


def detect_anomalies_rule_based(df, s1_rules, s2_action_map, bounds, device_type_dict, constant_devices,
                                frozen_sensors):
    print("🟣 [Engine B] 运行规则检测 (S1-S5)...")
    total_rows = len(df)
    anomaly_timeline = [[] for _ in range(total_rows)]

    def add_anomaly(start, end, message):
        safe_end = min(end, total_rows - 1)
        if start > safe_end: return
        for i in range(start, safe_end + 1): anomaly_timeline[i].append(message)

    robust_events = find_events_robust(df, device_type_dict)
    robust_events.sort(key=lambda x: x['idx'])

    print("   -> 标记水位状态...")
    tank_bool_series_map = {}
    target_sensors = [s for s in bounds.keys() if s in df.columns]
    for sensor in target_sensors:
        actual_min, actual_max = bounds[sensor]['lower'], bounds[sensor]['upper']
        high_trigger = actual_max - ((actual_max - actual_min) * THRESHOLD_PCT)
        low_trigger = actual_min + ((actual_max - actual_min) * THRESHOLD_PCT)
        series = pd.to_numeric(df[sensor], errors='coerce').fillna(method='ffill')
        trend = series.diff(TREND_WINDOW).fillna(0)
        tank_bool_series_map[f"{sensor} is High"] = (series >= high_trigger) & (trend > 0) & (
                    trend.abs() > FLUCTUATION_VAL)
        tank_bool_series_map[f"{sensor} is Low"] = (series <= low_trigger) & (trend < 0) & (
                    trend.abs() > FLUCTUATION_VAL)

    def check_range_window(device, target_val, start_idx, end_idx):
        if device not in df.columns: return False
        w_s, w_e = max(0, start_idx - SEARCH_BUFFER), min(total_rows, end_idx + SEARCH_BUFFER)
        return target_val in df[device].iloc[w_s:w_e].values

    def check_point_window(device, target_val, center_idx):
        if device not in df.columns: return False
        w_s, w_e = max(0, center_idx - SEARCH_BUFFER), min(total_rows, center_idx + SEARCH_BUFFER)
        return target_val in df[device].iloc[w_s:w_e].values

    # S1
    for cond_key, bool_series in tank_bool_series_map.items():
        if cond_key not in s1_rules: continue
        if not bool_series.any(): continue
        indices = bool_series[bool_series].index.values
        intervals = []
        if len(indices) > 0:
            s, e = indices[0], indices[0]
            for i in range(1, len(indices)):
                if indices[i] == e + 1:
                    e = indices[i]
                else:
                    intervals.append((s, e)); s, e = indices[i], indices[i]
            intervals.append((s, e))
        for (s, e) in intervals:
            first_missing = None
            for act in s1_rules[cond_key]:
                if not check_range_window(act['device'], act['target'], s, e):
                    first_missing = act;
                    break
            if first_missing:
                rec = find_end_s1(df, s, total_rows, first_missing['device'], first_missing['target'], bool_series)
                add_anomaly(s, rec - 1, f"S1: {cond_key}, Missing {first_missing['device']}")

    # S2 & S3
    active_s2 = set()
    for ev in robust_events:
        dev, state, idx = ev['device'], ev['state'], ev['idx']
        if dev in active_s2: active_s2.remove(dev); continue
        if (dev, state) not in s2_action_map: continue

        justified = None
        for rule in s2_action_map[(dev, state)]:
            cond = rule['condition']
            w_s, w_e = max(0, idx - SEARCH_BUFFER), min(total_rows, idx + SEARCH_BUFFER)
            if cond in tank_bool_series_map and tank_bool_series_map[cond].iloc[w_s:w_e].any():
                justified = rule;
                break

        if justified is None:
            next_chg = get_next_change_index(df, dev, idx, total_rows)
            add_anomaly(idx, next_chg - 1, f"S2: {dev} Unjustified")
            active_s2.add(dev)
        else:
            first_missing = None
            for p in justified['partners']:
                if not check_point_window(p['device'], p['target'], idx):
                    first_missing = p;
                    break
            if first_missing:
                rec = find_end_s3(df, idx, total_rows, dev, state, first_missing['device'], first_missing['target'])
                add_anomaly(idx, rec - 1, f"S3: {dev} Acted, but {first_missing['device']} Missing")

    # S4 & S5
    print("   -> 检测常值与冻结...")
    for grp in [constant_devices.get('constant_actuators', {}), constant_devices.get('constant_sensors', {})]:
        for dev, exp in grp.items():
            if dev not in df.columns: continue
            mask = pd.to_numeric(df[dev], errors='coerce').fillna(exp) != exp
            if mask.any():
                idxs = mask[mask].index.values
                if len(idxs) > 0:
                    s, e = idxs[0], idxs[0]
                    for i in range(1, len(idxs)):
                        if idxs[i] == e + 1:
                            e = idxs[i]
                        else:
                            add_anomaly(s, e, f"S4: Dev {dev}"); s, e = idxs[i], idxs[i]
                    add_anomaly(s, e, f"S4: Dev {dev}")

    if frozen_sensors:
        for sen, min_std in frozen_sensors.items():
            if sen not in df.columns: continue
            mask = df[sen].rolling(STD_CHECK_WINDOW).std().fillna(min_std) == 0
            if mask.any():
                idxs = mask[mask].index.values
                if len(idxs) > 0:
                    s, e = idxs[0], idxs[0]
                    for i in range(1, len(idxs)):
                        if idxs[i] == e + 1:
                            e = idxs[i]
                        else:
                            add_anomaly(s, e, f"S5: Frozen {sen}"); s, e = idxs[i], idxs[i]
                    add_anomaly(s, e, f"S5: Frozen {sen}")

    print(f"   -> 空隙填补 (Gap <= {MAX_GAP_SIZE_RULE})...")
    anom_idxs = [i for i, m in enumerate(anomaly_timeline) if m]
    if len(anom_idxs) > 1:
        anom_arr = np.array(anom_idxs)
        diffs = np.diff(anom_arr)
        fills = np.where((diffs > 1) & (diffs <= MAX_GAP_SIZE_RULE + 1))[0]
        for i in fills:
            s_f, e_f = anom_arr[i] + 1, anom_arr[i + 1] - 1
            last = anomaly_timeline[anom_arr[i]][-1]
            for k in range(s_f, e_f + 1): anomaly_timeline[k].append(f"{last} (Gap)")

    rule_res = pd.Series([""] * total_rows, index=df.index, dtype=object)
    rule_pred = np.ones(total_rows)
    for i, msgs in enumerate(anomaly_timeline):
        if msgs:
            rule_res[i] = "; ".join(sorted(list(set(msgs))))
            rule_pred[i] = -1
    return rule_pred, rule_res


# ==============================================================================
# 3. 主程序
# ==============================================================================
def main():
    # --- 新增：命令行参数解析 ---
    parser = argparse.ArgumentParser(description='WADI Integrated Detection with Data Truncation')
    parser.add_argument('--data_ratio', type=float, default=1.0,
                        help='Percentage of dataset to use: 0.3, 0.5, 0.7, 0.9, 1.0')
    parser.add_argument('--data_file', type=str, default=DEFAULT_CONFIG['data_file'],
                        help='Path to data file')
    parser.add_argument('--rules_file', type=str, default=DEFAULT_CONFIG['rules_file'],
                        help='Path to rules configuration file')
    parser.add_argument('--bounds_file', type=str, default=DEFAULT_CONFIG['bounds_file'],
                        help='Path to bounds configuration file')
    parser.add_argument('--constant_file', type=str, default=DEFAULT_CONFIG['constant_file'],
                        help='Path to constant devices configuration file')
    parser.add_argument('--std_json_file', type=str, default=DEFAULT_CONFIG['std_json_file'],
                        help='Path to sensor standard deviation configuration file')
    parser.add_argument('--output_file', type=str, default=DEFAULT_CONFIG['output_file'],
                        help='Path to output result file')
    parser.add_argument('--min_anomaly_length_model', type=int, default=DEFAULT_CONFIG['min_anomaly_length_model'],
                        help='Minimum anomaly length for model detection')
    parser.add_argument('--default_threshold', type=float, nargs=2, default=DEFAULT_CONFIG['default_threshold'],
                        help='Default threshold for model detection (lower, upper)')
    parser.add_argument('--trend_window', type=int, default=DEFAULT_CONFIG['trend_window'],
                        help='Trend window for rule detection')
    parser.add_argument('--fluctuation_val', type=float, default=DEFAULT_CONFIG['fluctuation_val'],
                        help='Fluctuation value for rule detection')
    parser.add_argument('--search_buffer', type=int, default=DEFAULT_CONFIG['search_buffer'],
                        help='Search buffer for rule detection')
    parser.add_argument('--threshold_pct', type=float, default=DEFAULT_CONFIG['threshold_pct'],
                        help='Threshold percentage for rule detection')
    parser.add_argument('--max_gap_size_rule', type=int, default=DEFAULT_CONFIG['max_gap_size_rule'],
                        help='Maximum gap size for rule detection')
    parser.add_argument('--std_check_window', type=int, default=DEFAULT_CONFIG['std_check_window'],
                        help='Standard deviation check window for rule detection')
    args = parser.parse_args()

    # 更新配置变量
    global DATA_FILE, RULES_FILE, BOUNDS_FILE, CONSTANT_FILE, STD_JSON_FILE, OUTPUT_FILE
    global MIN_ANOMALY_LENGTH_MODEL, DEFAULT_THRESHOLD, TREND_WINDOW, FLUCTUATION_VAL
    global SEARCH_BUFFER, THRESHOLD_PCT, MAX_GAP_SIZE_RULE, STD_CHECK_WINDOW

    DATA_FILE = args.data_file
    RULES_FILE = args.rules_file
    BOUNDS_FILE = args.bounds_file
    CONSTANT_FILE = args.constant_file
    STD_JSON_FILE = args.std_json_file
    OUTPUT_FILE = args.output_file
    MIN_ANOMALY_LENGTH_MODEL = args.min_anomaly_length_model
    DEFAULT_THRESHOLD = tuple(args.default_threshold)
    TREND_WINDOW = args.trend_window
    FLUCTUATION_VAL = args.fluctuation_val
    SEARCH_BUFFER = args.search_buffer
    THRESHOLD_PCT = args.threshold_pct
    MAX_GAP_SIZE_RULE = args.max_gap_size_rule
    STD_CHECK_WINDOW = args.std_check_window

    print(f"🚀 读取数据: {DATA_FILE}")
    if not os.path.exists(DATA_FILE): print("❌ 文件未找到"); return

    try:
        df = pd.read_csv(DATA_FILE, encoding='utf-8')
    except:
        df = pd.read_csv(DATA_FILE, encoding='gbk')
    df.columns = df.columns.str.strip()

    # --- 核心修改：数据集截断逻辑 ---
    original_len = len(df)
    if args.data_ratio < 1.0:
        new_size = int(original_len * args.data_ratio)
        print(f"✂️  数据截断激活：取前 {args.data_ratio * 100:.0f}% 数据 (从 {original_len} 行截断至 {new_size} 行)")
        df = df.iloc[:new_size].reset_index(drop=True)
    else:
        print(f"📊 使用全量数据集进行测试 (100%)")

    total_rows = len(df)

    label_col = df.columns[-1]
    y_true_raw = pd.to_numeric(df[label_col], errors='coerce').fillna(1).values
    # 统一转换: 1=Normal, -1=Attack -> 0=Normal, 1=Attack
    y_true_binary = np.where(y_true_raw == -1, 1, 0)

    sensor_df = df.iloc[:, 3:-1].apply(pd.to_numeric, errors='coerce').fillna(0)

    # 1. 运行引擎 A (Model)
    y_raw_model, r_model = detect_anomalies_model(sensor_df, MODELS, MODEL_THRESHOLDS)
    y_final_model = filter_anomalies_by_length(y_raw_model, MIN_ANOMALY_LENGTH_MODEL)

    # 2. 运行引擎 B (Rule) - 直接内存运行，不依赖文件
    # 加载配置
    const_dev = {}
    if os.path.exists(CONSTANT_FILE):
        with open(CONSTANT_FILE, 'r') as f: const_dev = json.load(f)
    frozen_sen = {}
    if os.path.exists(STD_JSON_FILE):
        with open(STD_JSON_FILE, 'r') as f: frozen_sen = json.load(f)
    if os.path.exists(BOUNDS_FILE):
        with open(BOUNDS_FILE, 'r') as f:
            bnds = json.load(f).get("bounds", {})
    else:
        bnds = {}

    s1, s2 = parse_rules_reversed(RULES_FILE)

    # 运行
    y_final_rule, r_rule = detect_anomalies_rule_based(df, s1, s2, bnds, DEVICE_TYPE, const_dev, frozen_sen)

    # 3. 合并
    print("🟢 [Fusion] 结果合并...")
    y_combined = np.where((y_final_model == -1) | (y_final_rule == -1), -1, 1)

    final_reasons = []
    for m, r in zip(r_model, r_rule):
        p = []
        if m: p.append(m)
        if r: p.append(r)
        final_reasons.append(" | ".join(p))

    # 4. 评估
    print("\n" + "=" * 60)
    print("📊 最终合并结果评估")
    print("=" * 60)

    y_pred_binary = np.where(y_combined == -1, 1, 0)

    TP = ((y_pred_binary == 1) & (y_true_binary == 1)).sum()
    TN = ((y_pred_binary == 0) & (y_true_binary == 0)).sum()
    FP = ((y_pred_binary == 1) & (y_true_binary == 0)).sum()
    FN = ((y_pred_binary == 0) & (y_true_binary == 1)).sum()

    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    # Calculate AUROC and AUPRC
    try:
        auroc = roc_auc_score(y_true_binary, y_pred_binary)
        auprc = average_precision_score(y_true_binary, y_pred_binary)
    except Exception as e:
        auroc = 0
        auprc = 0
        print(f"Warning: Could not calculate AUROC/AUPRC: {e}")

    print(f"TP: {TP} | FN: {FN}")
    print(f"FP: {FP} | TN: {TN}")
    print("-" * 30)
    print(f"Accuracy:  {acc:.4%}")
    print(f"Precision: {prec:.4%}")
    print(f"Recall:    {rec:.4%}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"📈 AUROC:     {auroc:.4f}")
    print(f"📉 AUPRC:     {auprc:.4f}")
    print("=" * 60)

    # 保存
    res_df = pd.DataFrame()
    if len(df.columns) > 2: res_df['Timestamp'] = df.iloc[:, 1]
    res_df['Label_True'] = y_true_raw
    res_df['Label_Combined'] = y_combined
    res_df['Label_Model'] = y_final_model
    res_df['Label_Rule'] = y_final_rule
    res_df['Anomaly_Causes'] = final_reasons
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()