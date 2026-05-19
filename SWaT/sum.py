import argparse
from typing import Dict, List, Tuple, Set, Optional, Iterable
# 务必在文件顶部导入 csv 库，用于控制保存格式
import csv
import json
import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


    # Fallback: create a dummy tqdm that does nothing
    def tqdm(iterable, *args, **kwargs):
        return iterable



# ==========================================
# 1. 全局配置 (默认值)
# ==========================================
INPUT_FILE = 'SWaT_Dataset_Attack_v0.csv'
OUTPUT_FILE = 'SWaT_Integrated_Detection_Result.csv'

# [配置 A] LIT 物理模型参数
WINDOW_LIT = 90
DELTA_T = 1.0
MIN_STD_THRESHOLD = 0.1

# ==========================================
# 2. 模型参数库
# ==========================================
LIT_PARAMS = {
    'LIT101': {'inflow': 'FIT101', 'outflow': 'FIT201', 'level': 'LIT101', 'A': 43.423, 'k_in': 8.365, 'k_out': 8.438,
               'lower': -9.8684, 'upper': 11.9908},
    'LIT301': {'inflow': 'FIT201', 'outflow': 'FIT301', 'level': 'LIT301', 'A': 40.289, 'k_in': 7.929, 'k_out': 7.908,
               'lower': -8.95368, 'upper': 11.0067},
    'LIT401': {'inflow': 'FIT301', 'outflow': 'FIT401', 'level': 'LIT401', 'mv_col': 'MV303', 'A': 42.924,
               'k_in': 7.933, 'k_out': 8.481, 'lower': -27.114987, 'upper': 13.5564}
}

PIT_PARAMS = {
    'DPIT301': {'x': 'FIT301', 'y': 'DPIT301', 'H': 2.0515, 'A': 2.9865, 'B': 2.2458, 'lower': -12.655653,
                'upper': 4.170975},
    'PIT501': {'x': 'FIT501', 'y': 'PIT501', 'H': 447.1338, 'A': 3.2535, 'B': -67.6548, 'lower': -8.402645,
               'upper': 4.263151},
    'PIT502': {'x': 'FIT502', 'y': 'PIT502', 'H': 90.2222, 'A': -137.7122, 'B': 53.2143, 'lower': -0.671974,
               'upper': 0.876602},
    'PIT503': {'x': 'FIT503', 'y': 'PIT503', 'H': -2467.237, 'A': 6624.7388, 'B': -4095.2421, 'lower': -5.215095,
               'upper': 3.761217},
}

AIT_STATS = {
    'AIT202': {'mean': -0.000000, 'std': 0.12943, 'sigma': 5.0},
    'AIT504': {'mean': 0.000000, 'std': 1.58371, 'sigma': 6.0},
    'AIT501': {'mean': 0.000000, 'std': 0.13492, 'sigma': 25.0},
}


# ==========================================
# 3. 辅助函数
# ==========================================
def get_feature(df, name, lag, ftype):
    if lag == 0:
        shifted = df[name]
    else:
        shifted = df[name].shift(lag)
    if ftype == 'L':
        return shifted
    elif ftype == 'S':
        return shifted ** 2
    elif ftype == 'D':
        return shifted.diff()
    return shifted


def detect_lit(df, config):
    """
    LIT 物理模型检测 (含连续过滤和原因细分)
    返回: (Final_Flags, Phys_Flags, Std_Flags)
    """
    fit_in = df[config['inflow']].fillna(0).to_numpy()
    fit_out = df[config['outflow']].fillna(0).to_numpy()
    lit = df[config['level']].fillna(method='ffill').fillna(0).to_numpy()
    if 'mv_col' in config and config['mv_col'] in df.columns:
        mv_status = df[config['mv_col']].to_numpy()
        fit_in = np.where(mv_status == 2, 0, fit_in)

    if len(df) < WINDOW_LIT:
        zeros = pd.Series([False] * len(df), index=df.index)
        return zeros, zeros, zeros

    # 1. 物理模型
    kernel = np.ones(WINDOW_LIT)
    sum_in = np.convolve(fit_in, kernel, mode="valid")
    sum_out = np.convolve(fit_out, kernel, mode="valid")
    lit_start = lit[:-WINDOW_LIT + 1]
    lit_end = lit[WINDOW_LIT - 1:]
    delta_real = lit_end - lit_start
    delta_pred = (DELTA_T / config['A']) * (config['k_in'] * sum_in - config['k_out'] * sum_out)
    residual = delta_real - delta_pred

    # 2. 标准差
    rolling_std = df[config['level']].rolling(window=WINDOW_LIT).std()
    std_valid = rolling_std.iloc[WINDOW_LIT - 1:].to_numpy()

    # 3. 原始判定
    mask_phys = (residual < config['lower']) | (residual > config['upper'])
    mask_std = std_valid < MIN_STD_THRESHOLD
    raw_anomaly = mask_phys | mask_std

    # 4. 连续异常过滤
    min_len = int(np.ceil(WINDOW_LIT / 10))
    diff = np.diff(np.concatenate(([False], raw_anomaly, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    filtered_anomaly = raw_anomaly.copy()
    for start, end in zip(starts, ends):
        if end - start < min_len:
            filtered_anomaly[start:end] = False

    # 5. 生成结果 Series (与过滤后的结果对齐)
    final_flags_arr = np.zeros(len(df), dtype=bool)
    final_phys_arr = np.zeros(len(df), dtype=bool)
    final_std_arr = np.zeros(len(df), dtype=bool)

    final_flags_arr[WINDOW_LIT - 1:] = filtered_anomaly
    final_phys_arr[WINDOW_LIT - 1:] = mask_phys & filtered_anomaly
    final_std_arr[WINDOW_LIT - 1:] = mask_std & filtered_anomaly

    return (
        pd.Series(final_flags_arr, index=df.index),
        pd.Series(final_phys_arr, index=df.index),
        pd.Series(final_std_arr, index=df.index)
    )


def detect_pit(df, config):
    x = df[config['x']]
    y = df[config['y']]
    calc = config['H'] + config['A'] * x + config['B'] * (x ** 2)
    delta = y - calc
    is_normal = delta.between(config['lower'], config['upper'], inclusive="both")
    return ~is_normal


def detect_ait_regression(df, target_name):
    pred = None
    if target_name == 'AIT202':
        pred = (8.7490 - 0.0679 * get_feature(df, 'FIT201', 0, 'L') + 0.0460 * get_feature(df, 'FIT201', 0,
                                                                                           'D') + 0.0287 * get_feature(
            df, 'AIT203', 0, 'D') + 0.0257 * get_feature(df, 'FIT201', 0, 'S') + 0.0162 * get_feature(df, 'P205', 0,
                                                                                                      'D') + 0.0146 * get_feature(
            df, 'P203', 0, 'D') - 0.0106 * get_feature(df, 'MV201', 0, 'L') - 0.0074 * get_feature(df, 'P203', 0,
                                                                                                   'S') - 0.0064 * get_feature(
            df, 'P101', 0, 'D') + 0.0061 * get_feature(df, 'MV201', 0, 'S') - 0.0051 * get_feature(df, 'P205', 0,
                                                                                                   'S') - 0.0025 * get_feature(
            df, 'P203', 0, 'L') - 0.0017 * get_feature(df, 'P205', 0, 'L') + 0.0014 * get_feature(df, 'P101', 0,
                                                                                                  'S') + 0.0006 * get_feature(
            df, 'MV201', 0, 'D') + 0.0005 * get_feature(df, 'P101', 0, 'L') + 0.0004 * get_feature(df, 'AIT203', 0,
                                                                                                   'L'))
    elif target_name == 'AIT504':
        pred = (450.0964 + 3.3383 * get_feature(df, 'FIT501', 0, 'S') + 2.4148 * get_feature(df, 'FIT401', 0,
                                                                                             'S') - 2.1037 * get_feature(
            df, 'PIT503', 0, 'L') - 2.0666 * get_feature(df, 'PIT501', 0, 'L') + 0.9416 * get_feature(df, 'FIT501', 0,
                                                                                                      'L') + 0.6781 * get_feature(
            df, 'FIT401', 0, 'L') + 0.4715 * get_feature(df, 'AIT402', 0, 'L') + 0.2861 * get_feature(df, 'PIT501', 0,
                                                                                                      'D') - 0.2278 * get_feature(
            df, 'PIT503', 0, 'D') - 0.1745 * get_feature(df, 'AIT502', 0, 'L') - 0.1413 * get_feature(df, 'FIT401', 0,
                                                                                                      'D') - 0.1194 * get_feature(
            df, 'FIT501', 0, 'D') + 0.0407 * get_feature(df, 'FIT503', 0, 'S') + 0.0335 * get_feature(df, 'FIT504', 0,
                                                                                                      'L') + 0.0296 * get_feature(
            df, 'AIT502', 0, 'D') + 0.0228 * get_feature(df, 'FIT504', 0, 'S') + 0.0212 * get_feature(df, 'FIT503', 0,
                                                                                                      'L') + 0.0068 * get_feature(
            df, 'PIT503', 0, 'S') - 0.0046 * get_feature(df, 'AIT402', 0, 'D') + 0.0030 * get_feature(df, 'PIT501', 0,
                                                                                                      'S') - 0.0023 * get_feature(
            df, 'FIT503', 0, 'D') - 0.0014 * get_feature(df, 'AIT402', 0, 'S') - 0.0007 * get_feature(df, 'FIT504', 0,
                                                                                                      'D') + 0.0004 * get_feature(
            df, 'AIT502', 0, 'S'))
    elif target_name == 'AIT501':
        pred = (-4.1765 + 0.0441 * get_feature(df, 'AIT201', 0, 'L') + 0.0432 * get_feature(df, 'AIT503', 0,
                                                                                            'L') - 0.0028 * get_feature(
            df, 'AIT503', 0, 'D') - 0.0014 * get_feature(df, 'AIT201', 0, 'D') - 0.0001 * get_feature(df, 'AIT503', 0,
                                                                                                      'S') - 0.0001 * get_feature(
            df, 'AIT201', 0, 'S'))

    residual = df[target_name] - pred
    stats = AIT_STATS[target_name]
    threshold = stats['sigma'] * stats['std']
    return (residual < (stats['mean'] - threshold)) | (residual > (stats['mean'] + threshold))


DEVICE_TYPE: Dict[str, str] = {
    'FIT101': 'Sensor', 'LIT101': 'Sensor', 'MV101': 'Actuator', 'P101': 'Actuator',
    'P102': 'Actuator',
    'AIT201': 'Sensor', 'AIT202': 'Sensor', 'AIT203': 'Sensor',
    'FIT201': 'Sensor', 'MV201': 'Actuator', 'P201': 'Actuator', 'P202': 'Actuator',
    'P203': 'Actuator', 'P204': 'Actuator', 'P205': 'Actuator', 'P206': 'Actuator',
    'DPIT301': 'Sensor', 'FIT301': 'Sensor', 'LIT301': 'Sensor', 'MV301': 'Actuator',
    'MV302': 'Actuator', 'MV303': 'Actuator', 'MV304': 'Actuator', 'P301': 'Actuator',
    'P302': 'Actuator',
    'AIT401': 'Sensor', 'AIT402': 'Sensor', 'FIT401': 'Sensor',
    'LIT401': 'Sensor',
    'P401': 'Actuator', 'P402': 'Actuator', 'P403': 'Actuator', 'P404': 'Actuator',
    'UV401': 'Actuator',
    'AIT501': 'Sensor', 'AIT502': 'Sensor', 'AIT503': 'Sensor', 'AIT504': 'Sensor',
    'FIT501': 'Sensor', 'FIT502': 'Sensor', 'FIT503': 'Sensor', 'FIT504': 'Sensor',
    'P501': 'Actuator', 'P502': 'Actuator',
    'PIT501': 'Sensor', 'PIT502': 'Sensor', 'PIT503': 'Sensor',
    'FIT601': 'Sensor',
    'P601': 'Actuator', 'P602': 'Actuator', 'P603': 'Actuator',
}

SENSOR_PREFIX_THRESHOLDS: Dict[str, float] = {
    'FIT': 0.4,
    'AIT': 0.3,
    'PIT': 0.3,
    'DPIT': 3.0,
    'LIT': 1.5,
}

# Backup pump mutual-exclusion pairs and equivalence
P_BACKUP_PAIRS: List[Tuple[str, str]] = [
    ('P101', 'P102'),
    ('P202', 'P203'),
    ('P301', 'P302'),
    ('P401', 'P402'),
    ('P404', 'P403'),
    ('P502', 'P501'),
]


def build_pump_equiv_map(pairs: Iterable[Tuple[str, str]]) -> Dict[str, Set[str]]:
    equiv: Dict[str, Set[str]] = {}
    for a, b in pairs:
        equiv.setdefault(a, set()).update([a, b])
        equiv.setdefault(b, set()).update([a, b])
    return equiv


PUMP_EQUIV_MAP = build_pump_equiv_map(P_BACKUP_PAIRS)

# Primary-backup pump pairs for opposite-direction suppression (primary, backup)
P_PRIMARY_BACKUP_PAIRS: List[Tuple[str, str]] = [
    ('P101', 'P102'),
    ('P201', 'P202'),
    ('P203', 'P204'),
    ('P205', 'P206'),
    ('P302', 'P301'),
    ('P402', 'P401'),
    ('P403', 'P404'),
    ('P501', 'P502'),
]

PRIMARY_PUMPS: Set[str] = {primary for (primary, _) in P_PRIMARY_BACKUP_PAIRS}
BACKUP_PUMPS: Set[str] = {backup for (_, backup) in P_PRIMARY_BACKUP_PAIRS}


def _label_direction(label: str) -> Optional[str]:
    if isinstance(label, str):
        if label.endswith('_opening_trend'):
            return 'opening'
        if label.endswith('_closing_trend'):
            return 'closing'
    return None


def _suppress_primary_backup_opposite(labels: List[str]) -> List[str]:
    """
    If a primary pump and its backup both have events at the same row
    and their directions are opposite (one opening, one closing),
    suppress events of both pumps for that pair in this row.

    This only removes the two pump events from this row (equivalent to skipping matching),
    and does NOT affect other devices' events in the same row.
    """
    if not labels:
        return labels
    to_remove: Set[str] = set()
    # Build tag -> label map (one per pump expected)
    # Only process P-type labels (pumps), other devices are not affected
    tag_to_label: Dict[str, str] = {}
    for lb in labels:
        tag = _get_event_tag(lb)
        if not tag:
            continue
        if tag.startswith('P'):
            tag_to_label[tag] = lb
    # Check each primary-backup pair for opposite directions
    for primary, backup in P_PRIMARY_BACKUP_PAIRS:
        lb_p = tag_to_label.get(primary)
        lb_b = tag_to_label.get(backup)
        if not lb_p or not lb_b:
            continue
        d_p = _label_direction(lb_p)
        d_b = _label_direction(lb_b)
        if d_p is None or d_b is None:
            continue
        # Opposite when one 'opening' and the other 'closing'
        if (d_p == 'opening' and d_b == 'closing') or (d_p == 'closing' and d_b == 'opening'):
            # Remove both pump events from this row
            to_remove.add(lb_p)
            to_remove.add(lb_b)
    if not to_remove:
        return labels
    # Return all labels except the removed pump events
    # Other devices' events remain unchanged
    return [lb for lb in labels if lb not in to_remove]


def detect_backup_pump_conflicts(df: pd.DataFrame, row_idx: int) -> List[str]:
    """
    Check if any backup pump pair is simultaneously ON (value==2) at row_idx.
    Returns list of pair tags that are in conflict, e.g., ['P101&P102'].
    """
    conflicts: List[str] = []
    for a, b in P_BACKUP_PAIRS:
        if a in df.columns and b in df.columns:
            va = pd.to_numeric(df[a], errors='coerce')
            vb = pd.to_numeric(df[b], errors='coerce')
            try:
                va_i = va.iloc[row_idx]
                vb_i = vb.iloc[row_idx]
            except Exception:
                continue
            if pd.notna(va_i) and pd.notna(vb_i) and int(va_i) == 2 and int(vb_i) == 2:
                conflicts.append(f"{a}&{b}")
    return conflicts


def _was_pump_already_on(df: pd.DataFrame, tag: str, row_idx: int) -> bool:
    """Check if pump was already ON (value==2) at row_idx-1."""
    if row_idx <= 0 or tag not in df.columns:
        return False
    try:
        s = pd.to_numeric(df[tag], errors='coerce')
        prev_val = s.iloc[row_idx - 1]
        if pd.notna(prev_val):
            return int(prev_val) == 2
    except Exception:
        pass
    return False


def _are_both_pumps_closed(df: pd.DataFrame, a: str, b: str, row_idx: int) -> bool:
    """Check if both pumps are closed (value==1) at row_idx."""
    if a not in df.columns or b not in df.columns:
        return False
    try:
        sa = pd.to_numeric(df[a], errors='coerce')
        sb = pd.to_numeric(df[b], errors='coerce')
        va = sa.iloc[row_idx]
        vb = sb.iloc[row_idx]
        if pd.notna(va) and pd.notna(vb):
            return int(va) == 1 and int(vb) == 1
    except Exception:
        pass
    return False


def expand_pump_equivalents(labels: List[str]) -> List[str]:
    """
    Expand P-type labels by replacing pump tag with equivalent backup pump tag labels.
    E.g., P101_opening_trend -> {P101_opening_trend, P102_opening_trend} when applicable.
    """
    expanded: Set[str] = set()
    for lab in labels:
        tag = _get_event_tag(lab)
        if not tag or not tag.startswith('P'):
            expanded.add(lab)
            continue
        eq = PUMP_EQUIV_MAP.get(tag)
        if not eq:
            expanded.add(lab)
            continue
        suffix = lab[len(tag):]
        for t in eq:
            expanded.add(f"{t}{suffix}")
    return list(expanded)


def _any_equiv_act_event_in_window(
        label_to_positions: Dict[str, List[int]],
        label: str,
        start_pos: int,
        end_pos: int,
) -> bool:
    """Return True if label or any equivalent pump label occurs in [start_pos, end_pos]."""
    tag = _get_event_tag(label)
    candidates: List[str]
    if tag and tag.startswith('P') and tag in PUMP_EQUIV_MAP:
        suffix = label[len(tag):]
        candidates = [f"{t}{suffix}" for t in PUMP_EQUIV_MAP[tag]]
    else:
        candidates = [label]
    for cand in candidates:
        if has_event_in_window(label_to_positions, cand, start_pos, end_pos):
            return True
    return False


def _first_equiv_act_event_pos_in_window(
        label_to_positions: Dict[str, List[int]],
        label: str,
        start_pos: int,
        end_pos: int,
) -> Optional[int]:
    """Return the first position of label (or any equivalent pump label) in [start_pos, end_pos]."""
    tag = _get_event_tag(label)
    candidates: List[str]
    if tag and tag.startswith('P') and tag in PUMP_EQUIV_MAP:
        suffix = label[len(tag):]
        candidates = [f"{t}{suffix}" for t in PUMP_EQUIV_MAP[tag]]
    else:
        candidates = [label]
    best: Optional[int] = None
    for cand in candidates:
        positions = label_to_positions.get(cand)
        if not positions:
            continue
        for p in positions:
            if p < start_pos:
                continue
            if p > end_pos:
                break
            if best is None or p < best:
                best = p
            break
    return best


def _first_equiv_act_event_pos_and_label_in_window(
        label_to_positions: Dict[str, List[int]],
        label: str,
        start_pos: int,
        end_pos: int,
) -> Optional[Tuple[int, str]]:
    """Return (first_position, actual_label) among label and its equivalent pump labels within [start_pos, end_pos]."""
    tag = _get_event_tag(label)
    candidates: List[str]
    if tag and tag.startswith('P') and tag in PUMP_EQUIV_MAP:
        suffix = label[len(tag):]
        candidates = [f"{t}{suffix}" for t in PUMP_EQUIV_MAP[tag]]
    else:
        candidates = [label]
    best: Optional[Tuple[int, str]] = None
    for cand in candidates:
        positions = label_to_positions.get(cand)
        if not positions:
            continue
        for p in positions:
            if p < start_pos:
                continue
            if p > end_pos:
                break
            if best is None or p < best[0]:
                best = (p, cand)
            break
    return best


def _all_equiv_act_event_pos_and_label_in_window(
        label_to_positions: Dict[str, List[int]],
        label: str,
        start_pos: int,
        end_pos: int,
) -> List[Tuple[int, str]]:
    """Return all (position, actual_label) among label and equivalent pump labels within [start_pos, end_pos], sorted by position."""
    tag = _get_event_tag(label)
    candidates: List[str]
    if tag and tag.startswith('P') and tag in PUMP_EQUIV_MAP:
        suffix = label[len(tag):]
        candidates = [f"{t}{suffix}" for t in PUMP_EQUIV_MAP[tag]]
    else:
        candidates = [label]
    results: List[Tuple[int, str]] = []
    for cand in candidates:
        positions = label_to_positions.get(cand)
        if not positions:
            continue
        for p in positions:
            if p < start_pos:
                continue
            if p > end_pos:
                break
            results.append((int(p), cand))
    # sort and deduplicate by (pos, label)
    if results:
        results = sorted(set(results), key=lambda x: x[0])
    return results


def infer_timestamp_column(df: pd.DataFrame) -> str:
    candidates = [
        'Timestamp', 'timestamp', ' Date Time', 'Date Time', 'Datetime', 'DateTime', 'Time', 'time', ' Timestamp'
    ]
    for name in candidates:
        if name in df.columns:
            return name
    # fallback: use the first column
    return df.columns[0]


def load_bounds(bounds_path: str) -> Dict[str, Dict[str, float]]:
    with open(bounds_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'bounds' in data:
        return data['bounds']
    return data


def get_prefix(tag: str) -> Optional[str]:
    # Match longest known prefixes first
    for prefix in sorted(SENSOR_PREFIX_THRESHOLDS.keys(), key=len, reverse=True):
        if tag.startswith(prefix):
            return prefix
    return None


def detect_mv_events(series: pd.Series, tag: str) -> Dict[int, str]:
    """
    Detect MV actuator events: record the first change to 0 as the event time.
    Classify as opening (1->0->2) or closing (2->0->1) when possible by peeking ahead.
    Returns mapping of row index to reason label string, e.g., "MV201_opening_trend".
    """
    s = pd.to_numeric(series, errors='coerce')
    prev = s.shift(1)
    at_zero = (s == 0)
    came_from_nonzero = prev.isin([1, 2])
    event_indices = s.index[at_zero & came_from_nonzero]

    events: Dict[int, str] = {}
    values = s.values
    idx_array = s.index.to_numpy()
    for idx in event_indices:
        pos = np.where(idx_array == idx)[0]
        if len(pos) == 0:
            continue
        i = int(pos[0])
        prev_val = prev.iloc[i]
        # Look ahead to find the next non-zero value
        next_nonzero_val: Optional[float] = None
        for j in range(i + 1, len(values)):
            v = values[j]
            if pd.notna(v) and v != 0:
                next_nonzero_val = float(v)
                break
        label_suffix = 'change_to_0'
        if pd.notna(prev_val) and next_nonzero_val is not None:
            if int(prev_val) == 1 and int(next_nonzero_val) == 2:
                label_suffix = 'opening_trend'
            elif int(prev_val) == 2 and int(next_nonzero_val) == 1:
                label_suffix = 'closing_trend'
        events[i] = f"{tag}_{label_suffix}"
    # Enforce minimum gap between opening and closing events (<3 rows are ignored)
    if events:
        items = sorted(events.items(), key=lambda x: x[0])
        filtered: Dict[int, str] = {}
        last_kept_pos: Optional[int] = None
        last_kept_type: Optional[str] = None  # 'opening' | 'closing' | 'other'
        for pos_i, label in items:
            if label.endswith('_opening_trend'):
                typ = 'opening'
            elif label.endswith('_closing_trend'):
                typ = 'closing'
            else:
                typ = 'other'

            if last_kept_pos is None or typ == 'other':
                filtered[pos_i] = label
                if typ != 'other':
                    last_kept_pos = pos_i
                    last_kept_type = typ
                continue

            if typ != 'other' and last_kept_type is not None and typ != last_kept_type and (pos_i - last_kept_pos) < 3:
                # skip this event due to insufficient gap
                continue

            filtered[pos_i] = label
            if typ != 'other':
                last_kept_pos = pos_i
                last_kept_type = typ

        return filtered
    return events


def detect_p_events(series: pd.Series, tag: str) -> Dict[int, str]:
    """
    Detect P actuator events: 1->2 opening, 2->1 closing; event time is the change time.
    Returns mapping of row index to reason label string, e.g., "P101_opening_trend".
    """
    s = pd.to_numeric(series, errors='coerce')
    prev = s.shift(1)
    open_idx = s.index[(prev == 1) & (s == 2)]
    close_idx = s.index[(prev == 2) & (s == 1)]

    events: Dict[int, str] = {}
    for i in open_idx:
        # map positional index, not label index, to align with row ordering
        events[s.index.get_loc(i)] = f"{tag}_opening_trend"
    for i in close_idx:
        events[s.index.get_loc(i)] = f"{tag}_closing_trend"
    # Enforce minimum gap between opening and closing events (<3 rows are ignored)
    if events:
        items = sorted(events.items(), key=lambda x: x[0])
        filtered: Dict[int, str] = {}
        last_kept_pos: Optional[int] = None
        last_kept_type: Optional[str] = None
        for pos_i, label in items:
            if label.endswith('_opening_trend'):
                typ = 'opening'
            elif label.endswith('_closing_trend'):
                typ = 'closing'
            else:
                typ = 'other'

            if last_kept_pos is None or typ == 'other':
                filtered[pos_i] = label
                if typ != 'other':
                    last_kept_pos = pos_i
                    last_kept_type = typ
                continue

            if typ != 'other' and last_kept_type is not None and typ != last_kept_type and (pos_i - last_kept_pos) < 3:
                continue

            filtered[pos_i] = label
            if typ != 'other':
                last_kept_pos = pos_i
                last_kept_type = typ

        return filtered
    return events


def detect_actuator_events(df: pd.DataFrame) -> Dict[int, List[str]]:
    """
    Build a mapping from row position to list of actuator event labels occurring at that row.
    """
    row_to_events: Dict[int, List[str]] = {}
    present_tags = [c for c in df.columns if c in DEVICE_TYPE and DEVICE_TYPE[c] == 'Actuator']

    for tag in present_tags:
        series = df[tag]
        if tag.startswith('MV'):
            events = detect_mv_events(series, tag)
        elif tag.startswith('P'):
            events = detect_p_events(series, tag)
        else:
            # Unknown actuator type; skip
            events = {}

        for row_pos, label in events.items():
            row_to_events.setdefault(row_pos, []).append(label)

    # Suppress primary-backup opposite-direction events per row
    if row_to_events:
        for pos, labels in list(row_to_events.items()):
            filtered = _suppress_primary_backup_opposite(labels)
            if filtered:
                row_to_events[pos] = filtered
            else:
                del row_to_events[pos]

    return row_to_events


def detect_sensor_antecedent_events(
        df: pd.DataFrame,
        bounds: Dict[str, Dict[str, float]],
        trend_window: int,
) -> Dict[int, List[str]]:
    """
    Sliding-window sensor antecedent events per rules:
    - If any value in the window reaches/exceeds the upper bound AND (end - start) > prefix_threshold -> event at window middle row.
    - If any value in the window reaches/under-runs the lower bound AND (start - end) > prefix_threshold -> event at window middle row.
    Returns mapping from row position (window middle) to list of reason labels per sensor.
    """
    row_to_events: Dict[int, List[str]] = {}

    # candidate sensor tags are those present both in DataFrame and bounds
    sensor_tags: List[str] = []
    for col in df.columns:
        if col in DEVICE_TYPE and DEVICE_TYPE[col] == 'Sensor' and col in bounds:
            # Skip FIT sensors per requirement
            if isinstance(col, str) and col.startswith('FIT'):
                continue
            sensor_tags.append(col)

    if trend_window <= 1:
        raise ValueError('trend_window must be >= 2')

    for tag in sensor_tags:
        series = pd.to_numeric(df[tag], errors='coerce')
        lower = bounds[tag]['lower']
        upper = bounds[tag]['upper']
        prefix = get_prefix(tag)
        if prefix is None or prefix not in SENSOR_PREFIX_THRESHOLDS:
            # Skip sensors without configured prefix threshold
            continue
        delta_threshold = SENSOR_PREFIX_THRESHOLDS[prefix]

        # Rolling check for upper/lower bound reach within window
        rolling_max = series.rolling(window=trend_window, min_periods=trend_window).max()
        rolling_min = series.rolling(window=trend_window, min_periods=trend_window).min()
        upper_reached = rolling_max >= upper
        lower_reached = rolling_min <= lower

        # Start and end values of each window ending at i
        start_vals = series.shift(trend_window - 1)
        end_vals = series

        # Delta conditions
        upward_delta_ok = (end_vals - start_vals) > delta_threshold
        downward_delta_ok = (start_vals - end_vals) > delta_threshold

        # Indices where events occur (by label index); convert to positional
        idx_upper = series.index[upper_reached & upward_delta_ok]
        idx_lower = series.index[lower_reached & downward_delta_ok]

        half = trend_window // 2
        n_rows = len(df.index)

        for idx in idx_upper:
            end_pos = df.index.get_loc(idx)
            mid_pos = max(0, min(n_rows - 1, end_pos - half))
            row_to_events.setdefault(mid_pos, []).append(f"{tag}_upper_trend")
        for idx in idx_lower:
            end_pos = df.index.get_loc(idx)
            mid_pos = max(0, min(n_rows - 1, end_pos - half))
            row_to_events.setdefault(mid_pos, []).append(f"{tag}_lower_trend")

    # Densify: for the same sensor reason label, if consecutive events are
    # within <= 3 rows apart, fill the gap rows with the same label
    if row_to_events:
        label_to_positions: Dict[str, List[int]] = {}
        for pos, labels in row_to_events.items():
            for label in set(labels):
                label_to_positions.setdefault(label, []).append(pos)

        for label, positions in label_to_positions.items():
            positions.sort()
            for a, b in zip(positions, positions[1:]):
                gap = b - a
                if 1 < gap <= 3:
                    for k in range(a + 1, b):
                        row_to_events.setdefault(k, []).append(label)

        # Deduplicate labels per row while preserving order
        for pos, labels in list(row_to_events.items()):
            seen: Set[str] = set()
            row_to_events[pos] = [x for x in labels if not (x in seen or seen.add(x))]

        # Filter: keep only sensor antecedent labels that form continuous runs > 3 rows
        # i.e., sequences of consecutive positions with length >= 4
        # Build label -> sorted positions again after densify
        label_to_positions = {}
        for pos, labels in row_to_events.items():
            for label in labels:
                label_to_positions.setdefault(label, []).append(pos)
        for label in list(label_to_positions.keys()):
            label_to_positions[label].sort()

        label_to_keep_positions: Dict[str, Set[int]] = {}
        MIN_RUN = 4
        for label, positions in label_to_positions.items():
            if not positions:
                continue
            run_start_idx = 0

            def add_run(start_i: int, end_i: int) -> None:
                # inclusive range of indices [start_i, end_i], positions are consecutive by construction
                if end_i >= start_i and (end_i - start_i + 1) >= MIN_RUN:
                    keep = label_to_keep_positions.setdefault(label, set())
                    for i in range(start_i, end_i + 1):
                        keep.add(positions[i])

            for i in range(1, len(positions)):
                if positions[i] != positions[i - 1] + 1:
                    add_run(run_start_idx, i - 1)
                    run_start_idx = i
            add_run(run_start_idx, len(positions) - 1)

        # Apply filtering to row_to_events
        for pos, labels in list(row_to_events.items()):
            new_labels: List[str] = []
            for label in labels:
                keep_set = label_to_keep_positions.get(label, set())
                if pos in keep_set:
                    new_labels.append(label)
            if new_labels:
                row_to_events[pos] = new_labels
            else:
                # remove the row entry if empty
                del row_to_events[pos]

    return row_to_events


def merge_events_to_output(
        timestamps: pd.Series,
        actuator_events: Dict[int, List[str]],
        sensor_events: Dict[int, List[str]],
) -> pd.DataFrame:
    n = len(timestamps)
    act_list: List[str] = [''] * n
    sens_list: List[str] = [''] * n

    for i, labels in actuator_events.items():
        if 0 <= i < n:
            # Deduplicate but keep stable order
            seen: Set[str] = set()
            ordered = [x for x in labels if not (x in seen or seen.add(x))]
            act_list[i] = ';'.join(ordered)

    for i, labels in sensor_events.items():
        if 0 <= i < n:
            seen: Set[str] = set()
            ordered = [x for x in labels if not (x in seen or seen.add(x))]
            sens_list[i] = ';'.join(ordered)

    out = pd.DataFrame({
        'timestamp': timestamps,
        'actuator_events': act_list,
        'sensor_antecedent_events': sens_list,
    })
    return out


# ----------------------------- Rule matching -----------------------------

def _split_semicolon_list(cell: str) -> List[str]:
    if not isinstance(cell, str) or cell.strip() == '':
        return []
    return [x.strip() for x in cell.split(';') if x.strip()]


def _get_event_tag(label: str) -> Optional[str]:
    if not isinstance(label, str):
        return None
    # Known suffixes
    suffixes = ['_opening_trend', '_closing_trend', '_change_to_0', '_upper_trend', '_lower_trend']
    for suf in suffixes:
        if label.endswith(suf):
            return label[: -len(suf)]
    # Fallback: take token until last underscore
    if '_' in label:
        return label.split('_')[0]
    return None


def _is_actuator_label(label: str) -> bool:
    tag = _get_event_tag(label)
    return bool(tag and tag in DEVICE_TYPE and DEVICE_TYPE[tag] == 'Actuator')


def _is_sensor_label(label: str) -> bool:
    tag = _get_event_tag(label)
    return bool(tag and tag in DEVICE_TYPE and DEVICE_TYPE[tag] == 'Sensor')


class Rule:
    def __init__(self, rule_id: str, source: str, antecedent: str, consequents: List[str],
                 context: Optional[str] = None):
        self.rule_id = rule_id
        self.source = source  # 'act-act' | 'pair'
        self.antecedent = antecedent
        self.consequents = list(consequents)
        self.context = (context or '').strip().lower() if isinstance(context, str) else None

    def __repr__(self) -> str:
        return f"Rule({self.source},{self.antecedent}->{self.consequents})"


def load_act_act_rules(path: str) -> List[Rule]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    rules: List[Rule] = []
    for i, row in df.iterrows():
        antecedent = str(row['antecedent']).strip()
        consequents_raw = row.get('consequent_events', '')
        if pd.isna(consequents_raw):
            consequents_raw = ''
        if isinstance(consequents_raw, str):
            # split by comma respecting that it may already be a simple list
            consequents = [x.strip() for x in consequents_raw.split(',') if x.strip()]
        else:
            consequents = []
        context = row.get('context', None)
        if antecedent:
            rules.append(Rule(rule_id=f"actact_{i}", source='act-act', antecedent=antecedent, consequents=consequents,
                              context=context))
    return rules


def _read_csv_with_encodings(path: str) -> pd.DataFrame:
    # Try utf-8 then gbk
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding='gbk')


def load_pair_rules(path: str) -> List[Rule]:
    if not os.path.exists(path):
        return []
    df = _read_csv_with_encodings(path)
    # Use first two columns as antecedent, consequent (headers may be localized)
    if df.shape[1] < 2:
        return []
    col_a = df.columns[0]
    col_b = df.columns[1]
    rules: List[Rule] = []
    for i, row in df.iterrows():
        antecedent = str(row[col_a]).strip()
        cons_raw = row.get(col_b, '')
        if pd.isna(cons_raw):
            cons_raw = ''
        if isinstance(cons_raw, str):
            consequents = [x.strip() for x in cons_raw.split(',') if x.strip()]
        else:
            try:
                consequents = [str(cons_raw).strip()] if str(cons_raw).strip() else []
            except Exception:
                consequents = []
        if antecedent and consequents:
            rules.append(Rule(rule_id=f"pair_{i}", source='pair', antecedent=antecedent, consequents=consequents))
    return rules


def build_rule_indexes(rules: List[Rule]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    antecedent_index: Dict[str, List[int]] = {}
    consequent_index: Dict[str, List[int]] = {}
    for idx, r in enumerate(rules):
        antecedent_index.setdefault(r.antecedent, []).append(idx)
        for c in r.consequents:
            consequent_index.setdefault(c, []).append(idx)
    return antecedent_index, consequent_index


def load_sensor_step_thresholds(path: str, multiplier: float = 2.0) -> Dict[str, float]:
    """读取正常数据集的单步最大变化绝对值，并乘以 multiplier 作为阈值。
    使用第1列（设备名）与第2列（最大单步变化绝对值）。
    返回 tag -> threshold_abs 的映射。
    """
    if not os.path.exists(path):
        return {}
    df = _read_csv_with_encodings(path)
    if df.shape[1] < 2:
        return {}
    col_tag = df.columns[0]
    col_max_step = df.columns[1]
    thresholds: Dict[str, float] = {}
    for _, row in df.iterrows():
        try:
            tag = str(row[col_tag]).strip()
        except Exception:
            continue
        if not tag:
            continue
        try:
            max_step = float(row[col_max_step])
        except Exception:
            continue
        thresholds[tag] = abs(max_step) * float(multiplier)
    return thresholds


def load_sensor_step_stats(path: str, multiplier: float = 3.0) -> Dict[str, Dict[str, float]]:
    """读取单步最大变化与历史范围，返回 tag -> {thr, min, max}。
    CSV 列：第1列 名称，第2列 最大单步绝对值，第3列 历史最小值，第4列 历史最大值。
    thr = 最大单步绝对值 * multiplier。
    """
    stats: Dict[str, Dict[str, float]] = {}
    if not os.path.exists(path):
        return stats
    df = _read_csv_with_encodings(path)
    if df.shape[1] < 4:
        return stats
    col_tag = df.columns[0]
    col_max_step = df.columns[1]
    col_min = df.columns[2]
    col_max = df.columns[3]
    for _, row in df.iterrows():
        try:
            tag = str(row[col_tag]).strip()
        except Exception:
            continue
        if not tag:
            continue
        try:
            max_step = float(row[col_max_step])
            vmin = float(row[col_min])
            vmax = float(row[col_max])
        except Exception:
            continue
        stats[tag] = {
            'thr': abs(max_step) * float(multiplier),
            'min': float(vmin),
            'max': float(vmax),
        }
    return stats


def detect_single_step_overlimit_events(
        df: pd.DataFrame,
        stats: Dict[str, Dict[str, float]],
        plateau_S: int,
) -> Tuple[Dict[int, List[str]], List[int], Dict[str, List[int]]]:
    """在攻击数据集中检测各传感器的单步超域事件。
    条件：|x_t - x_{t-1}| > thresholds[tag]
    返回：
      - row_to_tags: 行位置 -> 该行触发超域的传感器列表
      - positions_sorted: 触发超域的所有行位置（升序）
    """
    row_to_tags: Dict[int, List[str]] = {}
    positions_set_all: Set[int] = set()
    positions_by_tag_sets: Dict[str, Set[int]] = {}
    if not stats:
        return {}, []
    for tag, meta in stats.items():
        if tag not in df.columns:
            continue
        try:
            thr = float(meta.get('thr', float('nan')))
            hist_min = float(meta.get('min', float('nan')))
            hist_max = float(meta.get('max', float('nan')))
        except Exception:
            continue
        if not np.isfinite(thr) or not np.isfinite(hist_min) or not np.isfinite(hist_max):
            continue
        try:
            s = pd.to_numeric(df[tag], errors='coerce')
        except Exception:
            continue
        diff_abs = s.diff().abs()
        idxs_all = diff_abs.index[diff_abs > thr]
        for idx in idxs_all:
            try:
                pos = df.index.get_loc(idx)
            except Exception:
                continue
            # 收集所有“超过阈值”的行（用于确定结束行）
            positions_set_all.add(int(pos))
            positions_by_tag_sets.setdefault(tag, set()).add(int(pos))
            # 起点行判定：之后的 S 行是否保持同一数值（不检查 LIT101/LIT301）
            if isinstance(plateau_S, int) and plateau_S > 0:
                n = len(s)
                if pos + plateau_S <= n - 1:
                    # 使用 pos+1 作为参考值，检查 pos+1..pos+S 是否一致
                    ref = s.iloc[pos + 1]
                    ok = True
                    if pd.isna(ref):
                        ok = False
                    else:
                        for k in range(pos + 2, pos + 1 + plateau_S):
                            v = s.iloc[k]
                            if pd.isna(v):
                                ok = False
                                break
                            try:
                                if float(v) != float(ref):
                                    ok = False
                                    break
                            except Exception:
                                if v != ref:
                                    ok = False
                                    break
                    if ok:
                        # 条件1和条件2满足即可，不再检查条件3（历史范围）
                        row_to_tags.setdefault(int(pos), []).append(tag)
            else:
                # 若未设置有效的 S，则默认所有超阈值行均为起点（退化为旧逻辑）
                row_to_tags.setdefault(int(pos), []).append(tag)
    positions_sorted_all = sorted(positions_set_all)
    positions_by_tag_sorted: Dict[str, List[int]] = {t: sorted(ps) for t, ps in positions_by_tag_sets.items()}
    return row_to_tags, positions_sorted_all, positions_by_tag_sorted


def _value_within_relaxed_end_range(
        tag: str,
        value: float,
        stats: Dict[str, Dict[str, float]],
        relax_p: float,
) -> bool:
    """判断给定数值是否落在放宽后的历史范围。
    放宽量 = p * |hist_max - hist_min|；
    下限 = max(0, hist_min - 放宽量)；上限 = hist_max + 放宽量。
    若缺少统计或 value 非数，则返回 False（视为不在范围内）。
    """
    if tag not in stats:
        return False
    try:
        hist_min = float(stats[tag]['min'])
        hist_max = float(stats[tag]['max'])
        if not np.isfinite(hist_min) or not np.isfinite(hist_max):
            return False
        width = abs(hist_max - hist_min)
        expand = float(relax_p) * float(width)
        lower = max(0.0, hist_min - expand)
        upper = hist_max + expand
        if pd.isna(value):
            return False
        v = float(value)
        return (v >= lower) and (v <= upper)
    except Exception:
        return False


def _shorten_act_interval(
        intervals: List[Dict[str, object]],
        interval_end_rows: Optional[Set[int]],
        base_tag: str,
        new_end: int,
        contexts: Optional[List[Dict[str, object]]] = None,
) -> None:
    """
    Shorten actuator abnormal interval(s) (S2/S3) associated with the given actuator base tag,
    so that their end <= new_end.
    - S2: match by extracting base tag from anchor and comparing with base_tag
    - S3: match by tag == base_tag (missing consequent actuator tag, already base tag)
    If contexts is provided, also update or remove related contexts when intervals are shortened.
    """
    if not intervals:
        return
    try:
        new_end_int = int(new_end)
    except Exception:
        return
    for it in intervals:
        try:
            scen = str(it.get('scenario', ''))
            if scen == 'S2':
                anchor_full = str(it.get('anchor', ''))
                anchor_base = _get_event_tag(anchor_full) if anchor_full else None
                match = (anchor_base == base_tag)
            elif scen in ('S1', 'S3'):
                tag_in_interval = str(it.get('tag', ''))
                match = (tag_in_interval == base_tag)
            else:
                match = False
            if not match:
                continue
            start = int(it.get('start', new_end_int))
            old_end = int(it.get('end', new_end_int))
            if new_end_int < start:
                adjusted_end = start
            else:
                adjusted_end = min(old_end, new_end_int)
            if adjusted_end != old_end:
                it['end'] = adjusted_end
                if interval_end_rows is not None:
                    interval_end_rows.discard(old_end)
                    if adjusted_end >= start:  # 只有当调整后的结束行 >= 开始行时才添加
                        interval_end_rows.add(adjusted_end)
                # 如果提供了 contexts，更新或删除相关的 context
                if contexts is not None:
                    rule_id = str(it.get('rule_id', ''))
                    # 查找匹配的 context 并更新结束行，或者如果新结束行小于原结束行，则标记为需要删除
                    contexts_to_remove: List[int] = []
                    for idx, ctx in enumerate(contexts):
                        try:
                            ctx_rule_id = str(ctx.get('rule_id', ''))
                            ctx_scenario = str(ctx.get('scenario', ''))
                            ctx_start = int(ctx.get('start', -1))
                            ctx_end = int(ctx.get('end', -1))
                            # 匹配规则ID、场景和起始行
                            if (ctx_rule_id == rule_id and ctx_scenario == scen and
                                    ctx_start == start and ctx_end == old_end):
                                # 如果新结束行小于原结束行，标记为需要删除
                                if adjusted_end < old_end:
                                    contexts_to_remove.append(idx)
                                else:
                                    # 更新结束行
                                    ctx['end'] = adjusted_end
                        except Exception:
                            continue
                    # 从后往前删除，避免索引变化
                    for idx in reversed(contexts_to_remove):
                        if 0 <= idx < len(contexts):
                            contexts.pop(idx)
        except Exception:
            continue


def invert_label_to_positions(events_map: Dict[int, List[str]]) -> Dict[str, List[int]]:
    label_to_positions: Dict[str, List[int]] = {}
    for pos, labels in events_map.items():
        for label in labels:
            label_to_positions.setdefault(label, []).append(pos)
    for label in list(label_to_positions.keys()):
        label_to_positions[label].sort()
    return label_to_positions


def _normalize_label_str(val: object) -> str:
    try:
        s = str(val)
    except Exception:
        return ''
    # Remove spaces and lowercase to catch cases like 'A ttack'
    return ''.join(s.split()).lower()


def _are_both_on_at(df: pd.DataFrame, a: str, b: str, pos: int) -> bool:
    if pos < 0 or pos >= len(df.index):
        return False
    if a not in df.columns or b not in df.columns:
        return False
    sa = pd.to_numeric(df[a], errors='coerce')
    sb = pd.to_numeric(df[b], errors='coerce')
    try:
        va = sa.iloc[pos]
        vb = sb.iloc[pos]
    except Exception:
        return False
    if pd.isna(va) or pd.isna(vb):
        return False
    try:
        return int(va) == 2 and int(vb) == 2
    except Exception:
        return False


def _end_of_mutual_on(df: pd.DataFrame, a: str, b: str, start_pos: int) -> int:
    """
    Given that at start_pos both a and b are ON (==2), find the first position j >= start_pos
    where they are NOT both ON anymore; return j. If never changes, return last index.
    """
    n = len(df.index)
    if n == 0:
        return -1
    start_pos = max(0, min(n - 1, start_pos))
    for j in range(start_pos, n):
        if not _are_both_on_at(df, a, b, j):
            return j
    return n - 1


def _next_actuator_state_change_pos(df: pd.DataFrame, tag: str, start_pos: int) -> int:
    """
    Find the next position >= start_pos where the actuator's discrete state changes
    compared to the immediately previous non-NaN value. If none, return the last row index.
    """
    if tag not in df.columns:
        return len(df.index) - 1
    s = pd.to_numeric(df[tag], errors='coerce')
    n = len(s)
    if n == 0:
        return -1
    start_pos = max(0, min(n - 1, start_pos))
    # Scan forward; state change at j when s[j] != s[j-1]
    for j in range(start_pos + 1, n):
        prev_v = s.iloc[j - 1]
        cur_v = s.iloc[j]
        if pd.isna(prev_v) or pd.isna(cur_v):
            continue
        try:
            if float(cur_v) != float(prev_v):
                return j
        except Exception:
            # Fallback strict inequality
            if cur_v != prev_v:
                return j
    return n - 1


def _desired_state_from_act_label(label: str) -> Optional[int]:
    """
    Map actuator consequent event label to desired stable state:
    - *_opening_trend -> 2 (open)
    - *_closing_trend -> 1 (closed)
    Otherwise: None
    """
    if not isinstance(label, str):
        return None
    if label.endswith('_opening_trend'):
        return 2
    if label.endswith('_closing_trend'):
        return 1
    return None


def _actuator_in_desired_state(df: pd.DataFrame, act_label: str, row_idx: int) -> bool:
    """
    True if at row_idx the actuator targeted by act_label already satisfies the desired state.
    For pumps (P*), also consider equivalent backup pumps.
    """
    tag = _get_event_tag(act_label)
    desired = _desired_state_from_act_label(act_label)
    if tag is None or desired is None:
        return False
    candidates: List[str]
    if tag.startswith('P') and tag in PUMP_EQUIV_MAP:
        candidates = list(PUMP_EQUIV_MAP[tag])
    else:
        candidates = [tag]
    for t in candidates:
        if t not in df.columns:
            continue
        try:
            val = pd.to_numeric(df[t], errors='coerce').iloc[row_idx]
        except Exception:
            continue
        if pd.notna(val):
            try:
                if int(val) == int(desired):
                    return True
            except Exception:
                pass
    return False


def _actuator_in_desired_state_in_window(df: pd.DataFrame, act_label: str, start_pos: int, end_pos: int) -> bool:
    """
    Check if the actuator targeted by act_label is in the desired state at any position within [start_pos, end_pos].
    For opening_trend, desired state is 2 (open); for closing_trend, desired state is 1 (closed).
    For pumps (P*), also consider equivalent backup pumps.
    """
    tag = _get_event_tag(act_label)
    desired = _desired_state_from_act_label(act_label)
    if tag is None or desired is None:
        return False
    candidates: List[str]
    if tag.startswith('P') and tag in PUMP_EQUIV_MAP:
        candidates = list(PUMP_EQUIV_MAP[tag])
    else:
        candidates = [tag]
    n = len(df.index)
    start_pos = max(0, start_pos)
    end_pos = min(n - 1, end_pos)
    if start_pos > end_pos:
        return False
    for t in candidates:
        if t not in df.columns:
            continue
        try:
            s = pd.to_numeric(df[t], errors='coerce')
        except Exception:
            continue
        # Check if any position in the window has the desired state
        for pos in range(start_pos, end_pos + 1):
            try:
                val = s.iloc[pos]
                if pd.notna(val):
                    try:
                        if int(val) == int(desired):
                            return True
                    except Exception:
                        pass
            except Exception:
                continue
    return False


def has_event_in_window(label_to_positions: Dict[str, List[int]], label: str, start_pos: int, end_pos: int) -> bool:
    positions = label_to_positions.get(label)
    if not positions:
        return False
    # Binary search could be used; linear scan is fine for small lists
    # Early exit when positions exceed end_pos
    for p in positions:
        if p < start_pos:
            continue
        if p > end_pos:
            return False
        return True
    return False


def _any_act_event_same_tag_in_window(
        label_to_positions: Dict[str, List[int]],
        label: str,
        start_pos: int,
        end_pos: int,
) -> bool:
    """
    是否在窗口 [start_pos, end_pos] 内存在“同一基础标签”的任意执行器事件
    （不限于同一种方向/后缀），例如：
      - 目标 label = MV101_closing_trend
      - 若窗口内出现 MV101_opening_trend，则认为“该执行器已有事件”
    该判断用于：一旦窗口内已有该执行器的事件，但没有找到当前 rule 需要的具体事件，
    就不再退化为用“状态值”来判定（防止用状态值假性满足相反方向的后项）。
    """
    base = _get_event_tag(label)
    if not base:
        return False
    for lb, positions in label_to_positions.items():
        if _get_event_tag(lb) != base:
            continue
        if not positions:
            continue
        for p in positions:
            if p < start_pos:
                continue
            if p > end_pos:
                break
            return True
    return False


## removed: collect_mv_sensor_bounds (use standalone script instead)


def sensor_trend_in_window(
        df: pd.DataFrame,
        label: str,
        start_pos: int,
        end_pos: int,
        trend_window: int,
) -> bool:
    """
    Check if the given sensor trend label (e.g., LIT101_upper_trend or LIT101_lower_trend)
    occurs within [start_pos, end_pos] using only delta threshold per prefix.
    The sliding window must be fully inside [start_pos, end_pos].
    """
    tag = _get_event_tag(label)
    if not tag or tag not in df.columns:
        return False
    prefix = get_prefix(tag)
    if prefix is None or prefix not in SENSOR_PREFIX_THRESHOLDS:
        return False
    delta_threshold = SENSOR_PREFIX_THRESHOLDS[prefix]
    s = pd.to_numeric(df[tag], errors='coerce')
    n = len(s)
    if trend_window <= 1:
        return False
    start_pos = max(0, start_pos)
    end_pos = min(n - 1, end_pos)
    if start_pos > end_pos:
        return False

    want_upper = label.endswith('_upper_trend')
    want_lower = label.endswith('_lower_trend')
    if not (want_upper or want_lower):
        return False

    # Slide end index from start_pos to end_pos; window start must be >= start_pos
    for end_idx in range(start_pos, end_pos + 1):
        start_idx = end_idx - trend_window + 1
        if start_idx < start_pos:
            continue
        if start_idx < 0:
            continue
        if end_idx >= n:
            break
        start_val = s.iloc[start_idx]
        end_val = s.iloc[end_idx]
        if pd.isna(start_val) or pd.isna(end_val):
            continue
        delta = float(end_val) - float(start_val)
        if want_upper and delta > delta_threshold:
            return True
        if want_lower and (-delta) > delta_threshold:
            return True
    return False


def match_rules_for_row(
        df: pd.DataFrame,
        row_idx: int,
        actuator_labels: List[str],
        sensor_antecedent_labels: List[str],
        rules: List[Rule],
        ant_index: Dict[str, List[int]],
        cons_index: Dict[str, List[int]],
        label_to_act_positions: Dict[str, List[int]],
        label_to_sensor_ant_positions: Dict[str, List[int]],
        k: int,
        trend_window: int,
        backwash_window: int,
        intervals_out: Optional[List[Dict[str, object]]] = None,
        interval_end_rows_out: Optional[Set[int]] = None,
        contexts_out: Optional[List[Dict[str, object]]] = None,
        prefound_by_rule_anchor: Optional[Dict[Tuple[str, str], Set[str]]] = None,
        capture_targets: Optional[Set[Tuple[str, str]]] = None,
        found_positions_capture_out: Optional[Dict[Tuple[str, str], Set[int]]] = None,
        ok_positions_by_rule_label: Optional[Dict[Tuple[str, str], Set[int]]] = None,
        ok_rule_ids_out: Optional[Set[str]] = None,
        windows_capture_out: Optional[List[Tuple[int, int]]] = None,
        suppress_act_anchor_labels: Optional[Set[str]] = None,
        mx_end_suppressed_tags: Optional[Set[str]] = None,
        s2_s3_end_suppress_tags: Optional[Set[str]] = None,
        bp_end_suppressed_tags: Optional[Set[str]] = None,
) -> List[str]:
    n_rows = len(df.index)
    reasons: List[str] = []

    def evaluate_rule(anchor_label: str, rule: Rule, anchor_role: str) -> str:
        # 1. 检查直通 OK (优化性能)
        if ok_positions_by_rule_label is not None:
            key_pos = (rule.rule_id, anchor_label)
            rows = ok_positions_by_rule_label.get(key_pos)
            if rows is not None and row_idx in rows:
                if ok_rule_ids_out is not None: ok_rule_ids_out.add(rule.rule_id)
                return f"OK: {rule.antecedent} -> {','.join(rule.consequents)}"

        required: List[Tuple[str, str]] = []
        required_pos: Dict[str, str] = {}
        if anchor_role == 'antecedent':
            anchor_is_sensor = _is_sensor_label(anchor_label)
            for c in rule.consequents:
                if _is_actuator_label(c):
                    required.append((c, 'act'))
                    required_pos[c] = 'consequent'
                elif _is_sensor_label(c):
                    if not anchor_is_sensor:
                        required.append((c, 'sensor_cons'))
                        required_pos[c] = 'consequent'
        else:
            ant = rule.antecedent
            if _is_actuator_label(ant):
                required.append((ant, 'act'))
                required_pos[ant] = 'antecedent'
            elif _is_sensor_label(ant):
                required.append((ant, 'sensor_ant'))
                required_pos[ant] = 'antecedent'
            for c in rule.consequents:
                if c == anchor_label: continue
                if _is_actuator_label(c):
                    required.append((c, 'act'))
                    required_pos[c] = 'consequent'
                elif _is_sensor_label(c):
                    required.append((c, 'sensor_cons'))
                    required_pos[c] = 'consequent'

        found: Set[str] = set()
        seed_key = (rule.rule_id, anchor_label)
        seed_labels: Set[str] = set()
        if prefound_by_rule_anchor is not None and seed_key in prefound_by_rule_anchor:
            seed_labels = set(prefound_by_rule_anchor[seed_key])

        # [修改点 1] 初始化 max_found_pos 为当前行
        # 这个变量用于记录为了满足规则，我们在时间轴上最远搜索到了哪里
        max_found_pos = row_idx

        def _seed_satisfies(lbl: str) -> bool:
            if lbl in seed_labels: return True
            if _is_actuator_label(lbl):
                tag = _get_event_tag(lbl)
                if tag and tag.startswith('P') and tag in PUMP_EQUIV_MAP:
                    suffix = lbl[len(tag):]
                    for t in PUMP_EQUIV_MAP[tag]:
                        if f"{t}{suffix}" in seed_labels: return True
            return False

        prev_found_count = -1
        expand = 0
        rule_ok_met = False
        capture_mode = bool(capture_targets is not None and (rule.rule_id, anchor_label) in (capture_targets or set()))
        cap_window_start = row_idx + 1 if capture_mode else 0
        labels_after_anchor: Set[str] = set()
        if capture_mode:
            seen_anchor = False
            for c in rule.consequents:
                if c == anchor_label:
                    seen_anchor = True
                    continue
                if not seen_anchor: continue
                if _is_actuator_label(c): labels_after_anchor.add(c)

        while True:
            start = max(0, row_idx - (expand + 1) * k)
            end = min(n_rows - 1, row_idx + (expand + 1) * k)
            if windows_capture_out is not None:
                if capture_mode:
                    if cap_window_start <= end: windows_capture_out.append((int(cap_window_start), int(end)))
                else:
                    windows_capture_out.append((int(start), int(end)))

            if capture_mode and found_positions_capture_out is not None and cap_window_start <= end:
                any_new_hits = False
                for label, role in required:
                    if role != 'act': continue
                    if required_pos.get(label) != 'consequent': continue
                    if labels_after_anchor and (label not in labels_after_anchor): continue
                    all_hits_slice = _all_equiv_act_event_pos_and_label_in_window(label_to_act_positions, label,
                                                                                  cap_window_start, end)
                    if all_hits_slice:
                        any_new_hits = True

                        # [修改点 2] 在 Capture 模式下更新 max_found_pos
                        # 找到该标签在本窗口内的第一次出现位置
                        first_hit_pos = min(p for p, _ in all_hits_slice)
                        if first_hit_pos > max_found_pos:
                            max_found_pos = first_hit_pos

                        for pos_found, actual_label in all_hits_slice:
                            key_store = (rule.rule_id, actual_label)
                            found_positions_capture_out.setdefault(key_store, set()).add(int(pos_found))
                cap_window_start = end + 1
                if not any_new_hits: break

            for label, role in required:
                if _seed_satisfies(label): found.add(label)

            for label, role in required:
                if label in found: continue
                if role == 'act':
                    matched = False
                    has_equiv_event = False
                    if capture_targets is not None and (rule.rule_id, anchor_label) in capture_targets:
                        start_search = max(start, row_idx + 1)
                        if labels_after_anchor and (label not in labels_after_anchor):
                            all_hits = []
                        else:
                            all_hits = _all_equiv_act_event_pos_and_label_in_window(label_to_act_positions, label,
                                                                                    start_search, end)
                        if all_hits:
                            matched = True
                            has_equiv_event = True
                            found.add(label)

                            # [修改点 3] 更新 max_found_pos
                            first_hit_pos = min(p for p, _ in all_hits)
                            if first_hit_pos > max_found_pos:
                                max_found_pos = first_hit_pos

                            if found_positions_capture_out is not None:
                                for pos_found, actual_label in all_hits:
                                    key_store = (rule.rule_id, actual_label)
                                    found_positions_capture_out.setdefault(key_store, set()).add(int(pos_found))
                    if not matched:
                        res = _first_equiv_act_event_pos_and_label_in_window(label_to_act_positions, label, start, end)
                        if res is not None:
                            has_equiv_event = True
                            matched = True
                            pos_found, actual_label = res
                            found.add(label)

                            # [修改点 4] 常规搜索中更新 max_found_pos
                            if pos_found > max_found_pos:
                                max_found_pos = pos_found

                    has_any_same_tag_event = _any_act_event_same_tag_in_window(label_to_act_positions, label, start,
                                                                               end)
                    if not matched and (not has_equiv_event) and (not has_any_same_tag_event):
                        if _actuator_in_desired_state(df, label, row_idx):
                            matched = True
                            found.add(label)
                            # 状态满足视为当前时刻满足，max_found_pos 保持 row_idx 即可
                elif role == 'sensor_ant':
                    if has_event_in_window(label_to_sensor_ant_positions, label, start, end): found.add(label)
                elif role == 'sensor_cons':
                    if sensor_trend_in_window(df, label, start, end, trend_window): found.add(label)

            if len(found) == len(required):
                if capture_mode:
                    remain_all_met = all((lab in found) for lab in labels_after_anchor) if labels_after_anchor else True
                    if remain_all_met: rule_ok_met = True
                else:
                    if ok_rule_ids_out is not None: ok_rule_ids_out.add(rule.rule_id)
                    if intervals_out is not None:
                        s2_base_tags_to_shorten: Set[str] = set()
                        if _is_actuator_label(anchor_label):
                            tag_a = _get_event_tag(anchor_label)
                            if tag_a: s2_base_tags_to_shorten.add(tag_a)
                        if _is_actuator_label(rule.antecedent):
                            tag_ant = _get_event_tag(rule.antecedent)
                            if tag_ant: s2_base_tags_to_shorten.add(tag_ant)
                        for c in rule.consequents:
                            if _is_actuator_label(c):
                                tag_c = _get_event_tag(c)
                                if tag_c: s2_base_tags_to_shorten.add(tag_c)
                        if s2_base_tags_to_shorten:
                            # [关键修改] 使用 max_found_pos 作为结束位置，确保异常覆盖到最后一个满足条件的事件
                            final_end_pos = max(row_idx, max_found_pos)
                            for base_tag in s2_base_tags_to_shorten:
                                _shorten_act_interval(intervals_out, interval_end_rows_out, base_tag, final_end_pos,
                                                      contexts_out)
                    return f"OK: {rule.antecedent} -> {','.join(rule.consequents)}"

            if (capture_mode is False) and (len(found) == prev_found_count):
                missing = [lbl for (lbl, _) in required if lbl not in found]
                ant_missing = any((required_pos.get(lbl) == 'antecedent') for lbl in missing)
                meta = f"[anchor={anchor_label};role={anchor_role};rule={rule.rule_id};ant_missing={1 if ant_missing else 0}]"
                if intervals_out is not None:
                    first_missing_act = None
                    for lbl, role in required:
                        if lbl in found: continue
                        if role == 'act':
                            first_missing_act = lbl
                            break
                    if anchor_role == 'antecedent':
                        if first_missing_act and required_pos.get(
                                first_missing_act) == 'consequent' and _is_actuator_label(first_missing_act):
                            tag_m = _get_event_tag(first_missing_act)
                            if tag_m:
                                end_pos = _next_actuator_state_change_pos(df, tag_m, row_idx)
                                intervals_out.append(
                                    {'start': row_idx, 'end': end_pos, 'scenario': 'S1', 'anchor': anchor_label,
                                     'missing': first_missing_act, 'rule_id': rule.rule_id, 'anchor_role': anchor_role,
                                     'tag': tag_m})
                                if interval_end_rows_out is not None:
                                    appeared = _any_equiv_act_event_in_window(label_to_act_positions, first_missing_act,
                                                                              end_pos, end_pos)
                                    if not appeared: interval_end_rows_out.add(end_pos)
                                if contexts_out is not None:
                                    seed_init = set(found)
                                    seed_init.add(rule.antecedent)
                                    contexts_out.append(
                                        {'rule_id': rule.rule_id, 'scenario': 'S1', 'start': row_idx, 'end': end_pos,
                                         'expected_label': first_missing_act, 'anchor_role': anchor_role,
                                         'anchor_label': anchor_label, 'antecedent': rule.antecedent,
                                         'found_seed': set(seed_init)})
                    else:
                        anchor_is_act = _is_actuator_label(anchor_label)
                        if anchor_is_act and ant_missing:
                            tag_a = _get_event_tag(anchor_label)
                            if tag_a:
                                end_pos = _next_actuator_state_change_pos(df, tag_a, row_idx)
                                intervals_out.append(
                                    {'start': row_idx, 'end': end_pos, 'scenario': 'S2', 'anchor': anchor_label,
                                     'missing': rule.antecedent, 'rule_id': rule.rule_id, 'anchor_role': anchor_role,
                                     'tag': tag_a})
                                if interval_end_rows_out is not None: interval_end_rows_out.add(end_pos)
                        ant_satisfied = (rule.antecedent in found)
                        if ant_satisfied:
                            if first_missing_act and required_pos.get(
                                    first_missing_act) == 'consequent' and _is_actuator_label(first_missing_act):
                                tag_m = _get_event_tag(first_missing_act)
                                if tag_m:
                                    end_pos = _next_actuator_state_change_pos(df, tag_m, row_idx)
                                    intervals_out.append(
                                        {'start': row_idx, 'end': end_pos, 'scenario': 'S3', 'anchor': anchor_label,
                                         'missing': first_missing_act, 'rule_id': rule.rule_id,
                                         'anchor_role': anchor_role, 'tag': tag_m})
                                    if interval_end_rows_out is not None:
                                        appeared = _any_equiv_act_event_in_window(label_to_act_positions,
                                                                                  first_missing_act, end_pos, end_pos)
                                        if not appeared: interval_end_rows_out.add(end_pos)
                                    if contexts_out is not None:
                                        seed_init = set(found)
                                        seed_init.add(rule.antecedent)
                                        contexts_out.append(
                                            {'rule_id': rule.rule_id, 'scenario': 'S3', 'start': row_idx,
                                             'end': end_pos, 'expected_label': first_missing_act,
                                             'anchor_role': anchor_role, 'anchor_label': anchor_label,
                                             'antecedent': rule.antecedent, 'found_seed': set(seed_init)})
                capture_info = "[capture_mode]" if capture_mode else ""
                return f"MISS: {rule.antecedent} -> {','.join(rule.consequents)}; 缺失: {','.join(missing)}; {meta}{capture_info}"

            prev_found_count = len(found)
            expand += 1

            if k <= 0:
                return f"MISS: {rule.antecedent} -> {','.join(rule.consequents)}; (K=0)"

        if rule_ok_met:
            if ok_rule_ids_out is not None: ok_rule_ids_out.add(rule.rule_id)
            if intervals_out is not None:
                s2_base_tags_to_shorten: Set[str] = set()
                if _is_actuator_label(anchor_label):
                    tag_a = _get_event_tag(anchor_label)
                    if tag_a: s2_base_tags_to_shorten.add(tag_a)
                if _is_actuator_label(rule.antecedent):
                    tag_ant = _get_event_tag(rule.antecedent)
                    if tag_ant: s2_base_tags_to_shorten.add(tag_ant)
                for c in rule.consequents:
                    if _is_actuator_label(c):
                        tag_c = _get_event_tag(c)
                        if tag_c: s2_base_tags_to_shorten.add(tag_c)
                if s2_base_tags_to_shorten:
                    # [关键修改] Capture 模式下，也使用 max_found_pos 作为结束位置
                    final_end_pos = max(row_idx, max_found_pos)
                    for base_tag in s2_base_tags_to_shorten:
                        _shorten_act_interval(intervals_out, interval_end_rows_out, base_tag, final_end_pos,
                                              contexts_out)
            return f"OK: {rule.antecedent} -> {','.join(rule.consequents)}"
        else:
            missing = [lbl for (lbl, _) in required if lbl not in found]
            ant_missing = any((required_pos.get(lbl) == 'antecedent') for lbl in missing)
            meta = f"[anchor={anchor_label};role={anchor_role};rule={rule.rule_id};ant_missing={1 if ant_missing else 0}]"
            if capture_mode and anchor_role == 'consequent':
                anchor_is_act = _is_actuator_label(anchor_label)
                if anchor_is_act and ant_missing:
                    tag_a = _get_event_tag(anchor_label)
                    if tag_a and intervals_out is not None:
                        end_pos = _next_actuator_state_change_pos(df, tag_a, row_idx)
                        intervals_out.append(
                            {'start': row_idx, 'end': end_pos, 'scenario': 'S2', 'anchor': anchor_label,
                             'missing': rule.antecedent, 'rule_id': rule.rule_id, 'anchor_role': anchor_role,
                             'tag': tag_a})
                        if interval_end_rows_out is not None: interval_end_rows_out.add(end_pos)
            return f"MISS: {rule.antecedent} -> {','.join(rule.consequents)}; 缺失: {','.join(missing)}; {meta}"

    def is_in_backwash(center: int) -> bool:
        has_mv303 = 'MV303' in df.columns
        has_mv301 = 'MV301' in df.columns
        if not has_mv303 and not has_mv301: return False
        start = max(0, center - backwash_window)
        end = min(n_rows - 1, center + backwash_window)
        mv303_in = False
        mv301_in = False
        if has_mv303:
            s303 = pd.to_numeric(df['MV303'], errors='coerce')
            mv303_in = s303.iloc[start: end + 1].eq(2).any()
        if has_mv301:
            s301 = pd.to_numeric(df['MV301'], errors='coerce')
            mv301_in = s301.iloc[start: end + 1].eq(2).any()
        return bool(mv303_in or mv301_in)

    in_backwash = is_in_backwash(row_idx)
    suppressed_actuator_tags: Set[str] = set()
    conflicts = detect_backup_pump_conflicts(df, row_idx=row_idx)
    if conflicts:
        for pair in conflicts:
            tags = pair.split('&')
            if len(tags) == 2:
                a, b = tags[0], tags[1]
                was_a_on = _was_pump_already_on(df, a, row_idx)
                was_b_on = _was_pump_already_on(df, b, row_idx)
                if was_a_on and not was_b_on:
                    suppressed_actuator_tags.add(b)
                elif was_b_on and not was_a_on:
                    suppressed_actuator_tags.add(a)
                elif was_a_on and was_b_on:
                    suppressed_actuator_tags.update([a, b])
        a_b = conflicts[0].split('&')
        if len(a_b) == 2:
            a, b = a_b[0], a_b[1]
            end_pos = _end_of_mutual_on(df, a, b, row_idx)
            try:
                if 'intervals_out' in locals() and intervals_out is not None:
                    intervals_out.append({'start': row_idx, 'end': end_pos, 'scenario': 'MX', 'anchor': f"{a}&{b}",
                                          'missing': 'mutual_exclusion', 'rule_id': 'mutual_exclusion',
                                          'anchor_role': 'n/a', 'tag': f"{a}&{b}"})
                if interval_end_rows_out is not None: interval_end_rows_out.add(end_pos)
            except Exception:
                pass
        suppressed_info = []
        if suppressed_actuator_tags: suppressed_info.append(f"suppressed={','.join(sorted(suppressed_actuator_tags))}")
        conflict_msg = f"ANOMALY: backup pumps simultaneously ON -> {','.join(conflicts)}"
        if suppressed_info: conflict_msg += f" ({';'.join(suppressed_info)})"
        reasons.append(conflict_msg)

    backup_on_tags: List[str] = []
    for pump in BACKUP_PUMPS:
        if pump not in df.columns: continue
        try:
            series = pd.to_numeric(df[pump], errors='coerce')
            val = series.iloc[row_idx]
        except Exception:
            continue
        if pd.isna(val): continue
        try:
            if int(val) == 2: backup_on_tags.append(pump)
        except Exception:
            continue
    if backup_on_tags:
        if intervals_out is not None:
            for pump in backup_on_tags:
                end_pos_bp = _next_actuator_state_change_pos(df, pump, row_idx)
                intervals_out.append(
                    {'start': row_idx, 'end': end_pos_bp, 'scenario': 'BP', 'anchor': pump, 'missing': 'backup_pump_on',
                     'rule_id': 'backup_pump_on', 'anchor_role': 'n/a', 'tag': pump})
                if interval_end_rows_out is not None: interval_end_rows_out.add(end_pos_bp)
        reasons.extend([f"ANOMALY: backup pump ON -> {pump}" for pump in backup_on_tags])

    for s_label in sensor_antecedent_labels:
        for rule_idx in ant_index.get(s_label, []):
            r = rules[rule_idx]
            if in_backwash and ('LIT401' in r.antecedent or any('LIT401' in c for c in r.consequents)): continue
            reasons.append(evaluate_rule(s_label, r, anchor_role='antecedent'))

    def _anchor_is_suppressed(label: str) -> bool:
        base = _get_event_tag(label)
        if base and base in suppressed_actuator_tags: return True
        if base and mx_end_suppressed_tags and base in mx_end_suppressed_tags: return True
        if base and s2_s3_end_suppress_tags and base in s2_s3_end_suppress_tags: return True
        if base and bp_end_suppressed_tags and base in bp_end_suppressed_tags: return True
        return bool(suppress_act_anchor_labels and label in suppress_act_anchor_labels)

    base_actuator_anchors = [a for a in actuator_labels if not _anchor_is_suppressed(a)]
    for a_label in expand_pump_equivalents(base_actuator_anchors):
        miss_recorded_for_anchor = False
        for rule_idx in cons_index.get(a_label, []):
            r = rules[rule_idx]
            if r.context == 'backwash' and not in_backwash: continue
            if in_backwash and ('LIT401' in r.antecedent or any('LIT401' in c for c in r.consequents)): continue
            res_text = evaluate_rule(a_label, r, anchor_role='consequent')
            reasons.append(res_text)
            if isinstance(res_text, str) and res_text.startswith('MISS:'):
                miss_recorded_for_anchor = True
                break
        if miss_recorded_for_anchor: continue
        for rule_idx in ant_index.get(a_label, []):
            r = rules[rule_idx]
            if r.context == 'backwash' and not in_backwash: continue
            if in_backwash and ('LIT401' in r.antecedent or any('LIT401' in c for c in r.consequents)): continue
            reasons.append(evaluate_rule(a_label, r, anchor_role='antecedent'))
    return reasons


def main():
    parser = argparse.ArgumentParser(description='SWaT Integrated Detection (Rule-based + Physical Models)')

    # --- 新增：数据集截断比例 ---
    parser.add_argument('--data_ratio', type=float, default=1.0,
                        help='Percentage of dataset to use: 0.3, 0.5, 0.7, 0.9, 1.0')

    # --- Logic A: 规则检测参数 ---
    parser.add_argument('--input', default='SWaT_Dataset_Attack_v0.csv', help='Input CSV file path')
    parser.add_argument('--rules', default='rules_summary.json', help='JSON file with bounds per sensor')
    parser.add_argument('--trend_window', type=int, default=10,
                        help='Sliding window size for sensor antecedent detection')
    parser.add_argument('--k', type=int, default=20, help='Search half-window (rows) for rule matching')
    parser.add_argument('--rules_act_act', default='rules_merged.csv',
                        help='CSV with actuator->(actuator|sensor) consequent rules')
    parser.add_argument('--rules_pairs', default='rules_summary_pairs.csv',
                        help='CSV with sensor-antecedent -> actuator consequent rules')
    parser.add_argument('--backwash_window', type=int, default=60,
                        help='Half-window for detecting backwash context (MV303==2)')
    parser.add_argument('--input_encoding', default='utf-8-sig', help='Encoding for input CSV file')
    parser.add_argument('--output_with_reason', default='detected_events_with_reason.csv',
                        help='Output CSV with rule matching')
    parser.add_argument('--output_encoding', default='utf-8-sig', help='Encoding for CSV outputs')

    # [修改] 已删除 sensor_step_normal, step_plateau_S, step_end_relax_p 三个参数

    # --- Logic B: 物理模型检测参数 ---
    parser.add_argument('--window_lit', type=int, default=90, help='Window size for LIT physical model')
    parser.add_argument('--output', type=str, default='SWaT_Integrated_Detection_Result.csv',
                        help='Final integrated output file')

    args = parser.parse_args()

    # 声明我们要修改这两个全局变量
    global MIN_STD_THRESHOLD, WINDOW_LIT, INPUT_FILE, OUTPUT_FILE

    # =======================================================
    # [原有] 根据 trend_window 调整 SENSOR 阈值
    # =======================================================
    scale_ratio = args.trend_window / 40.0
    print(f"\n🔧 正在根据 trend_window={args.trend_window} 调整 SENSOR阈值 (倍率={scale_ratio:.2f})...")
    for key in SENSOR_PREFIX_THRESHOLDS:
        SENSOR_PREFIX_THRESHOLDS[key] *= scale_ratio

    # =======================================================
    # [新增] 根据 window_lit 调整 MIN_STD_THRESHOLD
    # =======================================================
    # 逻辑：当前阈值 = 0.1 * (输入的window_lit / 90)
    original_std_thr = 0.1
    std_scale_ratio = args.window_lit / 90.0

    MIN_STD_THRESHOLD = original_std_thr * std_scale_ratio

    print(f"🔧 正在根据 window_lit={args.window_lit} 调整 MIN_STD_THRESHOLD (基准=90, 倍率={std_scale_ratio:.2f})")
    print(f"   -> 调整结果: {original_std_thr} -> {MIN_STD_THRESHOLD:.5f}")
    # =======================================================

    WINDOW_LIT = args.window_lit
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output

    # ==========================================
    # 1. 数据读取、清洗与【截断】
    # ==========================================
    if not os.path.exists(args.input):
        raise FileNotFoundError(f'Input CSV not found: {args.input}')

    print(f"正在读取 CSV 文件: {args.input}")
    try:
        df = pd.read_csv(args.input, encoding=args.input_encoding)
    except Exception:
        df = pd.read_csv(args.input, encoding='gbk')

    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    ts_col = infer_timestamp_column(df)

    # 强力清洗逻辑
    ts_str = df[ts_col].astype(str).str.strip()
    valid_mask = ts_str.str.match(r'^\d', na=False)
    df = df[valid_mask].reset_index(drop=True)

    # --- 核心修改：执行数据截断 ---
    if args.data_ratio < 1.0:
        new_size = int(len(df) * args.data_ratio)
        print(f"✂️  检测到截断请求：取数据集前 {args.data_ratio * 100:.0f}% (共 {new_size} 行)")
        df = df.iloc[:new_size].reset_index(drop=True)
    else:
        print(f"📊 使用全量数据集进行测试 (100%)")

    print(f"✓ 有效处理数据 {len(df)} 行")
    timestamps = df[ts_col]

    # ==========================================
    # 2. Logic A: 规则检测
    # ==========================================
    print("\n[Phase 1] 运行基于规则的检测逻辑...")
    bounds = load_bounds(args.rules)
    actuator_map = detect_actuator_events(df)
    sensor_map = detect_sensor_antecedent_events(df, bounds=bounds, trend_window=args.trend_window)

    out = merge_events_to_output(timestamps=timestamps, actuator_events=actuator_map, sensor_events=sensor_map)
    all_rules = load_act_act_rules(args.rules_act_act) + load_pair_rules(args.rules_pairs)
    ant_index, cons_index = build_rule_indexes(all_rules)
    label_to_act_positions = invert_label_to_positions(actuator_map)
    label_to_sensor_ant_positions = invert_label_to_positions(sensor_map)

    reasons_col: List[str] = []
    intervals: List[Dict[str, object]] = []
    interval_end_rows: Set[int] = set()
    contexts: List[Dict[str, object]] = []
    ok_positions_by_rule_label: Dict[Tuple[str, str], Set[int]] = {}
    search_windows_col: List[str] = [''] * len(out)

    print(f"  - 开始逐行规则匹配（共 {len(out)} 行）...")
    if HAS_TQDM:
        row_range = tqdm(range(len(out)), desc="Rule Matching", unit="row", ncols=100)
    else:
        row_range = range(len(out))

    for i in row_range:
        # Context handling
        prefound_map = {}
        end_row_rule_ids = set()
        capture_targets = set()
        found_positions_capture = {}

        if contexts:
            for ctx in contexts:
                try:
                    end_i = int(ctx.get('end', -1))
                    if end_i != i: continue
                    rule_id = str(ctx.get('rule_id', ''))
                    scenario = str(ctx.get('scenario', ''))
                    start_i = int(ctx.get('start', -1))
                    interval_still_valid = False
                    if intervals:
                        for it in intervals:
                            try:
                                if (str(it.get('rule_id', '')) == rule_id and str(
                                        it.get('scenario', '')) == scenario and
                                        int(it.get('start', -1)) == start_i and int(it.get('end', -1)) == i):
                                    interval_still_valid = True
                                    break
                            except Exception:
                                continue
                    if not interval_still_valid: continue

                    expected_label = str(ctx.get('expected_label', ''))
                    anchor_label_ctx = str(ctx.get('anchor_label', ''))
                    found_seed = set(ctx.get('found_seed', set()))

                    if _any_equiv_act_event_in_window(label_to_act_positions, expected_label, i, i):
                        key1 = (rule_id, expected_label)
                        prefound_map.setdefault(key1, set()).update(found_seed)
                        capture_targets.add((rule_id, expected_label))
                        if anchor_label_ctx:
                            prefound_map.setdefault((rule_id, anchor_label_ctx), set()).update(found_seed)
                        end_row_rule_ids.add(rule_id)
                except Exception:
                    pass

        pre_reasons_for_row: List[str] = []
        mx_end_suppressed_tags: Set[str] = set()

        if i in interval_end_rows and intervals:
            try:
                for it in intervals:
                    if int(it.get('end', -1)) == i and str(it.get('scenario', '')) == 'MX':
                        tag_pair = str(it.get('tag', ''))
                        if '&' in tag_pair:
                            tags = tag_pair.split('&')
                            if len(tags) == 2:
                                a, b = tags[0], tags[1]
                                if not _are_both_pumps_closed(df, a, b, i):
                                    try:
                                        va = pd.to_numeric(df[a], errors='coerce').iloc[i]
                                        vb = pd.to_numeric(df[b], errors='coerce').iloc[i]
                                        if pd.notna(va) and pd.notna(vb):
                                            if int(va) == 1 and int(vb) == 2:
                                                mx_end_suppressed_tags.add(a)
                                            elif int(vb) == 1 and int(va) == 2:
                                                mx_end_suppressed_tags.add(b)
                                    except Exception:
                                        pass
            except Exception:
                pass

        if i in interval_end_rows:
            # [修改] 删除了 SS 放宽检测逻辑 (allowed_start_tags 等)

            # 保留 Interval End 提示生成和去重逻辑
            detail_list = []
            if intervals:
                for it in intervals:
                    if int(it.get('end', -1)) == i:
                        scen = str(it.get('scenario', ''))
                        missing = str(it.get('missing', ''))
                        if missing:
                            detail_list.append(f"{scen} 缺失: {missing}")
                        else:
                            detail_list.append(scen)

            seen_details = set()
            unique_details = [x for x in detail_list if not (x in seen_details or seen_details.add(x))]
            detail_text = f" -> {' | '.join(unique_details)}" if unique_details else ""
            pre_reasons_for_row.append(f"ANOMALY: interval_end{detail_text}")

        act_labels = _split_semicolon_list(out.loc[i, 'actuator_events'])
        sensor_ant_labels = _split_semicolon_list(out.loc[i, 'sensor_antecedent_events'])

        s2_suppress_labels = set()
        if intervals:
            for it in intervals:
                if str(it.get('scenario', '')) == 'S2':
                    if int(it.get('start', -1)) <= i <= int(it.get('end', -1)):
                        lbl = str(it.get('anchor', ''))
                        if lbl: s2_suppress_labels.add(lbl)

        s2_s3_end_suppress_tags = set()
        if i in interval_end_rows and intervals:
            for it in intervals:
                scen = str(it.get('scenario', ''))
                if scen in ('S2', 'S3') and int(it.get('end', -1)) == i:
                    tg = str(it.get('tag', ''))
                    if tg: s2_s3_end_suppress_tags.add(tg)

        bp_end_suppressed_tags = set()
        if i in interval_end_rows and intervals:
            try:
                for it in intervals:
                    if int(it.get('end', -1)) == i and str(it.get('scenario', '')) in ('BP', 'MX'):
                        tag = str(it.get('tag', ''))
                        if tag:
                            if '&' in tag:
                                for t in tag.split('&'): bp_end_suppressed_tags.add(t)
                            else:
                                bp_end_suppressed_tags.add(tag)
            except Exception:
                pass

        ok_rules_this_row = set()
        windows_capture_this_row = []

        reasons = match_rules_for_row(
            df=df, row_idx=i, actuator_labels=act_labels, sensor_antecedent_labels=sensor_ant_labels,
            rules=all_rules, ant_index=ant_index, cons_index=cons_index,
            label_to_act_positions=label_to_act_positions,
            label_to_sensor_ant_positions=label_to_sensor_ant_positions,
            k=args.k, trend_window=args.trend_window, backwash_window=args.backwash_window,
            intervals_out=intervals, interval_end_rows_out=interval_end_rows, contexts_out=contexts,
            prefound_by_rule_anchor=prefound_map, capture_targets=capture_targets,
            found_positions_capture_out=found_positions_capture, ok_rule_ids_out=ok_rules_this_row,
            ok_positions_by_rule_label=ok_positions_by_rule_label, windows_capture_out=windows_capture_this_row,
            # [修改] 移除了 step_over_map 等参数的传入
            suppress_act_anchor_labels=s2_suppress_labels,
            mx_end_suppressed_tags=mx_end_suppressed_tags,
            s2_s3_end_suppress_tags=s2_s3_end_suppress_tags,
            bp_end_suppressed_tags=bp_end_suppressed_tags
        )

        full_reason = ';'.join(pre_reasons_for_row + reasons) if pre_reasons_for_row else ';'.join(reasons)
        cleaned_reason = full_reason.replace(',', '|').replace('\n', ' ').replace('\r', ' ')
        if len(cleaned_reason) > 32000:
            cleaned_reason = cleaned_reason[:32000] + "...(TRUNCATED)"
        reasons_col.append(cleaned_reason)

        if windows_capture_this_row:
            search_windows_col[i] = ';'.join([f"[{s}|{e}]" for (s, e) in windows_capture_this_row])

        if end_row_rule_ids and contexts:
            for ctx in contexts:
                try:
                    if int(ctx.get('end', -1)) != i: continue
                    rule_id = str(ctx.get('rule_id', ''))
                    if rule_id not in ok_rules_this_row: continue
                    if found_positions_capture:
                        for (rid, lab), poses in found_positions_capture.items():
                            if rid == rule_id and poses:
                                ok_positions_by_rule_label.setdefault((rid, lab), set()).update(poses)
                except Exception:
                    pass

    # 2.4 构建规则结果
    out_with_reason = out.copy()
    out_with_reason['reason'] = reasons_col
    out_with_reason['search_windows'] = search_windows_col

    pred_rule = np.zeros(len(out), dtype=int)
    if intervals:
        for it in intervals:
            s = max(0, int(it['start']))
            e = min(len(out) - 1, int(it['end']))
            if s <= e:
                pred_rule[s: e + 1] = 1

    out_with_reason['predict'] = pred_rule

    direct_ok_col = [''] * len(out)
    if ok_positions_by_rule_label:
        rule_text_map = {r.rule_id: f"{r.antecedent}->{'|'.join(r.consequents)}" for r in all_rules}
        row_to_rules_ok = {}
        for (rid, lab), poses in ok_positions_by_rule_label.items():
            for pos in poses:
                if 0 <= int(pos) < len(out):
                    row_to_rules_ok.setdefault(int(pos), []).append(rule_text_map.get(str(rid), str(rid)))
        for pos, rules_list in row_to_rules_ok.items():
            seen = set()
            ordered = [x for x in rules_list if not (x in seen or seen.add(x))]
            direct_ok_col[pos] = ';'.join(ordered)
    out_with_reason['direct_ok'] = direct_ok_col

    # ==========================================
    # 3. Logic B: 物理模型检测
    # ==========================================
    print("\n[Phase 2] 运行物理模型检测 (LIT/PIT/AIT)...")

    out_with_reason['reason'] = out_with_reason['reason'].fillna('')
    out_with_reason['predict_phys'] = 0

    lit_iter = LIT_PARAMS.items()
    if HAS_TQDM: lit_iter = tqdm(lit_iter, desc="[LIT] Models", unit="sensor", ncols=100)
    for name, config in lit_iter:
        if config['level'] in df.columns:
            flags, phys_flags, std_flags = detect_lit(df, config)
            out_with_reason[f'Flag_LIT_{name}'] = flags.astype(int)
            out_with_reason.loc[flags, 'predict_phys'] = 1
            if phys_flags.any():
                out_with_reason.loc[phys_flags, 'reason'] = out_with_reason.loc[phys_flags, 'reason'].apply(
                    lambda x: x + (";" if x else "") + f"LIT_{name}_PHYSICS"
                )
            if std_flags.any():
                out_with_reason.loc[std_flags, 'reason'] = out_with_reason.loc[std_flags, 'reason'].apply(
                    lambda x: x + (";" if x else "") + f"LIT_{name}_LOW_STD"
                )

    pit_iter = PIT_PARAMS.items()
    if HAS_TQDM: pit_iter = tqdm(pit_iter, desc="[PIT] Models", unit="sensor", ncols=100)
    for name, config in pit_iter:
        if config['x'] in df.columns and config['y'] in df.columns:
            flags = detect_pit(df, config)
            out_with_reason[f'Flag_PIT_{name}'] = flags.astype(int)
            out_with_reason.loc[flags, 'predict_phys'] = 1
            if flags.any():
                out_with_reason.loc[flags, 'reason'] = out_with_reason.loc[flags, 'reason'].apply(
                    lambda x: x + (";" if x else "") + f"PIT_{name}_MODEL"
                )

    ait_keys = list(AIT_STATS.keys())
    ait_iter = tqdm(ait_keys, desc="[AIT] Regression", ncols=100) if HAS_TQDM else ait_keys
    for name in ait_iter:
        if name in df.columns:
            flags = detect_ait_regression(df, name).fillna(False)
            out_with_reason[f'Flag_AIT_Reg_{name}'] = flags.astype(int)
            out_with_reason.loc[flags, 'predict_phys'] = 1
            if flags.any():
                out_with_reason.loc[flags, 'reason'] = out_with_reason.loc[flags, 'reason'].apply(
                    lambda x: x + (";" if x else "") + f"AIT_{name}_REGRESSION"
                )

    out_with_reason['reason'] = out_with_reason['reason'].str.strip(';')
    out_with_reason['reason'] = out_with_reason['reason'].str.replace('\n', ' ').str.replace('\r', ' ')

    # ==========================================
    # 4. 整合结果
    # ==========================================
    print("\n[Phase 3] 整合结果与保存...")

    final_pred = (out_with_reason['predict'] | out_with_reason['predict_phys']).astype(int)
    out_with_reason['Final_Prediction'] = final_pred

    last_col = df.columns[-1]
    truth_vals = df[last_col].apply(_normalize_label_str)
    truth = truth_vals.isin({'attack'}).astype(int).to_numpy()

    if len(truth) != len(out_with_reason):
        print(f"Warning: Aligning truth length {len(truth)} to result length {len(out_with_reason)}")
        m = min(len(truth), len(out_with_reason))
        truth = truth[:m]
        out_with_reason = out_with_reason.iloc[:m]
        final_pred = final_pred[:m]

    out_with_reason['true'] = truth

    print(f"保存详细结果: {args.output_with_reason}")
    out_with_reason.to_csv(args.output_with_reason, index=False, encoding=args.output_encoding, quoting=csv.QUOTE_ALL)

    print(f"保存整合结果: {args.output}")
    out_with_reason.to_csv(args.output, index=False, encoding=args.output_encoding, quoting=csv.QUOTE_ALL)

    if len(truth) > 0:
        y_true = truth
        y_pred = final_pred.values
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = FP / (TN + FP) if (TN + FP) > 0 else 0
        fnr = FN / (TP + FN) if (TP + FN) > 0 else 0

        # [新增] 在这里加入 AUROC 和 AUPRC 计算
        try:
            # 注意：基于二值预测计算
            auroc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auroc = 0.0  # 防止测试集只有一类样本导致报错

        try:
            auprc = average_precision_score(y_true, y_pred)
        except ValueError:
            auprc = 0.0

        print("=" * 60)
        print("📊 最终性能指标:")
        print(f"   - 混淆矩阵: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
        print(f"   - Accuracy : {accuracy:.4%}")
        print(f"   - Precision: {precision:.4%}")
        print(f"   - Recall   : {recall:.4%}")
        print(f"   - F1 Score : {f1:.4f}")
        print(f"   - FPR      : {fpr:.4%}")
        print(f"   - FNR      : {fnr:.4%}")

        # [新增] 在这里加入打印代码
        print(f"   - AUROC    : {auroc:.4f}")
        print(f"   - AUPRC    : {auprc:.4f}")
        print("=" * 60)
    else:
        print("No ground-truth labels available.")

if __name__ == '__main__':
    main()


