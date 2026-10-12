"""Predict a CSV/XLSX of new projects using a saved fitted pipeline."""
import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from data_utils import ROOT, ID, read_prediction_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--scenario', choices=['early', 'detailed'], default='early')
    parser.add_argument('--model', choices=['best', 'RandomForest', 'XGBoost', 'GLM_Gamma', 'Stacking'], default='best')
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'new_predictions.csv')
    args = parser.parse_args()
    manifest = json.loads((args.artifacts / 'manifest.json').read_text(encoding='utf-8'))
    scene = manifest['scenarios'][args.scenario]
    name = scene['recommended_model'] if args.model == 'best' else args.model
    bundle = joblib.load(args.artifacts / scene['models'][name]['file'])
    data = read_prediction_file(args.input)
    missing = sorted(set(bundle['features']) - set(data.columns))
    if missing:
        raise ValueError(f'缺少模型必需的列：{missing}')
    X = data[bundle['features']].copy()
    notes = [[] for _ in range(len(X))]
    for col in bundle['numeric_features']:
        converted = pd.to_numeric(X[col], errors='coerce')
        invalid = X[col].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f'{col} 包含无法解析的数值，输入行：{(np.flatnonzero(invalid)+2).tolist()}')
        if np.isinf(converted.to_numpy(dtype=float)).any():
            raise ValueError(f'{col} 包含无穷值')
        X[col] = converted
        low, high = bundle['training_feature_ranges'][col]
        for i in np.flatnonzero(converted.isna()):
            notes[i].append(f'{col}缺失，采用训练中位数')
        for i in np.flatnonzero((converted < low) | (converted > high)):
            notes[i].append(f'{col}超出训练范围[{low:g}, {high:g}]')
    values = bundle['model'].predict(X)
    if not np.isfinite(values).all():
        raise ValueError('预测结果包含无效数值')
    for i in np.flatnonzero(values <= 0):
        notes[i].append('预测非正，需复核；未自动截断')
    result = data.copy()
    result['预测单方造价'] = values
    result['预测场景'] = args.scenario
    result['模型'] = name
    result['输入检查'] = ['；'.join(row) for row in notes]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'{len(result)} 条预测已保存：{args.output.resolve()}，模型：{name}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
