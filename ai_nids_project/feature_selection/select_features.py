import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

DATASETS = [
    (os.path.join(PROCESSED_DIR, 'unsw_processed.csv'), os.path.join(PROCESSED_DIR, 'unsw_selected_features.csv')),
    (os.path.join(PROCESSED_DIR, 'cicids_processed.csv'), os.path.join(PROCESSED_DIR, 'cicids_selected_features.csv'))
]
TOP_N = 30  # modestly increase features for richer models
MAX_ROWS = int(os.environ.get('MAX_ROWS', '250000'))


def select_top_features(data_path, output_path, top_n=TOP_N):
    print(f"Processing {data_path} ...")
    df = pd.read_csv(data_path)
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42)
        print(f"[INFO] Sampled down to {MAX_ROWS} rows for feature selection")
    X = df.drop('label', axis=1)
    y = df['label']
    non_numeric = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric:
        print(f"Encoding non-numeric columns: {non_numeric}")
        X_encoded = pd.get_dummies(X)
    else:
        X_encoded = X
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_encoded, y)
    importances = rf.feature_importances_
    feature_names = X_encoded.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    top_features = feature_importance_df['Feature'][:top_n].tolist()
    if non_numeric:
        df_full_encoded = pd.get_dummies(df.drop('label', axis=1))
        # Ensure missing dummy columns are added
        for col in feature_names:
            if col not in df_full_encoded.columns:
                df_full_encoded[col] = 0
        df_selected = pd.concat([df_full_encoded[top_features], df['label']], axis=1)
    else:
        df_selected = df[top_features + ['label']]
    df_selected.to_csv(output_path, index=False)
    # Save feature names JSON for evaluation alignment
    feat_json_path = os.path.join(os.path.dirname(output_path), os.path.splitext(os.path.basename(output_path))[0].replace('_selected_features','') + '_feature_names.json')
    with open(feat_json_path, 'w', encoding='utf-8') as f:
        json.dump(top_features, f, indent=2)
    print(f"Top {top_n} features saved -> {output_path}\nFeature list -> {feat_json_path}")
    return top_features


def main():
    for data_path, output_path in DATASETS:
        if os.path.exists(data_path):
            select_top_features(data_path, output_path)
        else:
            print(f"WARNING: Missing dataset {data_path}, skipping.")


if __name__ == '__main__':
    main()
