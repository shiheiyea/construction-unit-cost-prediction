"""Read the two-level Excel header without changing the source workbook."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'Material' / '工作簿1.xlsx'
TARGET = '单方造价'
ID = '项目名称'


def load_training_data(path=SOURCE):
    raw = pd.read_excel(path, sheet_name='表1', header=None)
    headers = [str(raw.iloc[0, i] if i < 20 else raw.iloc[1, i]).strip()
               for i in range(raw.shape[1])]
    if len(set(headers)) != len(headers):
        raise ValueError('存在重复列名，请检查表头')
    data = raw.iloc[2:].copy()
    data.columns = headers
    data.insert(0, 'Excel行号', data.index + 1)
    for col in headers[1:]:
        numeric = pd.to_numeric(data[col], errors='coerce')
        invalid = data[col].notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f'{col} 含无法解析的数字，Excel行号：{data.loc[invalid, "Excel行号"].tolist()}')
        data[col] = numeric
    missing_target = data[TARGET].isna()
    excluded = data.loc[missing_target, ['Excel行号', ID]].to_dict('records')
    valid = data.loc[~missing_target].reset_index(drop=True)
    if not np.isfinite(valid[headers[1:]].to_numpy(dtype=float)[~valid[headers[1:]].isna().to_numpy()]).all():
        raise ValueError('数据包含无穷值')
    if (valid[TARGET] <= 0).any():
        raise ValueError('Gamma GLM 要求单方造价大于零')
    area_error = valid[headers[2]] - valid[headers[3]] - valid[headers[4]]
    issues = [{'Excel行号': int(valid.loc[i, 'Excel行号']), ID: valid.loc[i, ID],
               '总面积减地上地下面积': float(area_error[i])}
              for i in valid.index if abs(area_error[i]) > 0.01]
    audit = {'source': str(Path(path).resolve()), 'sheet': '表1', 'valid_samples': len(valid),
             'excluded_without_target': excluded, 'duplicate_rows': int(valid[headers].duplicated().sum()),
             'numeric_features': len(headers) - 2, 'missing_predictors': valid[headers[2:]].isna().sum().to_dict(),
             'area_issues_preserved': issues, 'target_unit': '假定元/m²，原表单方造价列未标单位',
             'unused_sheets': ['Sheet1', 'Sheet2'],
             'categorical_note': '项目名称仅为标识；结构和基础形式等已经独热编码。经济指标零值保留。'}
    return valid, {'early': headers[2:20], 'detailed': headers[2:]}, audit


def read_prediction_file(path):
    path = Path(path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    return pd.read_csv(path, encoding='utf-8-sig')
