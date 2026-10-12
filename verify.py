"""Verify CV accounting, saved model outputs and preprocessing behaviour."""
import hashlib
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from train import preprocessor
from data_utils import ROOT, SOURCE, TARGET, ID, load_training_data


def main():
    out = ROOT / 'outputs'
    data, features, audit = load_training_data()
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == manifest['source_sha256']
    assert len(data) == 24 and len(audit['excluded_without_target']) == 1
    scores = pd.read_csv(out / 'metrics.csv')
    oof = pd.read_csv(out / 'oof_predictions.csv')
    assert len(scores) == 10 and len(oof) == 240
    for _, row in scores.iterrows():
        p = oof[(oof.scenario == row.scenario) & (oof.model == row.model)]
        assert len(p) == 24 and p['Excel行号'].nunique() == 24
        assert p.groupby('fold').size().tolist() == [4] * 6
        np.testing.assert_allclose(row.MAE, mean_absolute_error(p.actual, p.predicted))
        np.testing.assert_allclose(row.RMSE, np.sqrt(mean_squared_error(p.actual, p.predicted)))
        np.testing.assert_allclose(row.R2, r2_score(p.actual, p.predicted))
        np.testing.assert_allclose(row.MAPE_percent, (abs(p.actual - p.predicted) / p.actual).mean() * 100)
    ref = oof[(oof.scenario == 'early') & (oof.model == 'MeanBaseline')].sort_values('Excel行号')
    for scene, cols in features.items():
        assert len(cols) == (18 if scene == 'early' else 43)
        assert TARGET not in cols and ID not in cols
        same = oof[(oof.scenario == scene) & (oof.model == 'MeanBaseline')].sort_values('Excel行号')
        np.testing.assert_array_equal(ref.fold, same.fold)
        for model, entry in manifest['scenarios'][scene]['models'].items():
            bundle = joblib.load(out / entry['file'])
            assert bundle['features'] == cols
            pred = bundle['model'].predict(data[cols])
            assert len(pred) == 24 and np.isfinite(pred).all()
            if model == 'GLM_Gamma':
                assert (pred > 0).all()
    # A validation value of 1000 must not affect means fitted on [1, 3].
    tiny = pd.DataFrame({'面积': [1., 3.], '结构': ['框架', '框剪']})
    prep = preprocessor(tiny).fit(tiny)
    probe = pd.DataFrame({'面积': [1000., np.nan], '结构': ['新类别', '框架']})
    transformed = prep.transform(probe)
    np.testing.assert_allclose(prep.named_transformers_['numeric'].named_steps['scaler'].mean_, [2.])
    np.testing.assert_allclose(transformed[:, 0], [998., 0.])
    assert np.isfinite(transformed).all()
    assert transformed[0, 1:].sum() == 0  # unseen category: ignored rather than error
    print('验证通过：240 条折外预测、相同划分、指标重算、8 个模型加载、Gamma 正值、标准化隔离、未知类别和缺失值处理。')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
