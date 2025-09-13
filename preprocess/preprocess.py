"""Unified preprocessing script with callable functions.
Replaces hard-coded absolute paths with project-relative paths.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import json
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MAX_ROWS = int(os.environ.get('MAX_ROWS', '250000'))
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def _stratified_limit(df: pd.DataFrame, label_col: str = 'label', max_rows: int = MAX_ROWS) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    sampled = df.groupby(label_col, group_keys=False).apply(
        lambda g: g.sample(max(1, int(len(g) * frac)), random_state=42)
    )
    if len(sampled) > max_rows:
        sampled = sampled.sample(max_rows, random_state=42)
    elif len(sampled) < max_rows:  # pad if rounding undershoots
        deficit = max_rows - len(sampled)
        pad = df.sample(deficit, replace=True, random_state=42)
        sampled = pd.concat([sampled, pad], ignore_index=True)
    return sampled


def preprocess_unsw(train_file='UNSW_NB15_training-set.csv', test_file='UNSW_NB15_testing-set.csv', output='unsw_processed.csv'):
    train_path = os.path.join(RAW_DIR, train_file)
    test_path = os.path.join(RAW_DIR, test_file)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)
    df.columns = df.columns.str.strip().str.lower()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'attack_cat' not in df.columns:
        raise ValueError("UNSW-NB15 missing 'attack_cat' column.")
    df['attack_cat'] = df['attack_cat'].fillna('Normal')
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['attack_cat'])
    # Persist label mapping (index -> original string)
    mapping = {
        'classes': list(map(str, label_encoder.classes_)),
        'index_to_label': {str(i): str(lbl) for i, lbl in enumerate(label_encoder.classes_)},
        'label_to_index': {str(lbl): int(i) for i, lbl in enumerate(label_encoder.classes_)}
    }
    mapping_path = os.path.join(MODEL_DIR, 'unsw_label_mapping.json')
    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2)
        print(f"[UNSW] Label mapping saved -> {mapping_path}")
    except Exception as e:
        print(f"[UNSW][WARN] Failed to save label mapping: {e}")
    df = df.drop(columns=['attack_cat'])
    df = df.fillna(0)
    num_cols = [c for c in df.select_dtypes(include=['number']).columns if c != 'label']
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    # Persist scaler
    scaler_path = os.path.join(MODEL_DIR, 'unsw_scaler.joblib')
    try:
        joblib.dump(scaler, scaler_path)
        print(f"[UNSW] Scaler saved -> {scaler_path}")
    except Exception as e:
        print(f"[UNSW][WARN] Failed to save scaler: {e}")
    original = len(df)
    df = _stratified_limit(df, 'label', MAX_ROWS)
    if len(df) < original:
        print(f"[UNSW] Reduced rows {original} -> {len(df)} (MAX_ROWS={MAX_ROWS})")
    out_path = os.path.join(PROCESSED_DIR, output)
    df.to_csv(out_path, index=False)
    print(f"UNSW-NB15 preprocessing complete -> {out_path}")
    return out_path


def preprocess_cicids(sub_dir='MachineLearningCSV (1)/MachineLearningCVE', output='cicids_processed.csv'):
    data_dir = os.path.join(RAW_DIR, sub_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"CICIDS directory not found: {data_dir}")
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.csv', '.xlsx'))]
    frames = []
    for f in files:
        if f.lower().endswith('.xlsx'):
            frames.append(pd.read_excel(f))
        else:
            frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(axis=1, how='all')
    df.columns = df.columns.str.strip().str.lower()
    if 'label' not in df.columns:
        raise ValueError("CICIDS2017: No 'label' column found!")
    df['label'] = df['label'].replace({'BENIGN': 'Normal'})
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    mapping = {
        'classes': list(map(str, label_encoder.classes_)),
        'index_to_label': {str(i): str(lbl) for i, lbl in enumerate(label_encoder.classes_)},
        'label_to_index': {str(lbl): int(i) for i, lbl in enumerate(label_encoder.classes_)}
    }
    mapping_path = os.path.join(MODEL_DIR, 'cicids_label_mapping.json')
    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2)
        print(f"[CICIDS] Label mapping saved -> {mapping_path}")
    except Exception as e:
        print(f"[CICIDS][WARN] Failed to save label mapping: {e}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)
    num_cols = [c for c in df.select_dtypes(include=['number']).columns if c != 'label']
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    scaler_path = os.path.join(MODEL_DIR, 'cicids_scaler.joblib')
    try:
        joblib.dump(scaler, scaler_path)
        print(f"[CICIDS] Scaler saved -> {scaler_path}")
    except Exception as e:
        print(f"[CICIDS][WARN] Failed to save scaler: {e}")
    original = len(df)
    df = _stratified_limit(df, 'label', MAX_ROWS)
    if len(df) < original:
        print(f"[CICIDS] Reduced rows {original} -> {len(df)} (MAX_ROWS={MAX_ROWS})")
    out_path = os.path.join(PROCESSED_DIR, output)
    df.to_csv(out_path, index=False)
    print(f"CICIDS2017 preprocessing complete -> {out_path}")
    return out_path


def main():
    preprocess_unsw()
    preprocess_cicids()


if __name__ == '__main__':
    main()

