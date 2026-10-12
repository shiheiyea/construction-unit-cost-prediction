"""Two prediction scenarios; nested CV with all preprocessing inside folds."""
import argparse
import hashlib
import html
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor
from data_utils import ROOT, SOURCE, TARGET, ID, load_training_data

SEED = 20261012
SCENARIOS = {'early': '项目特征', 'detailed': '项目特征＋分项价格'}


def preprocessor(X):
    nums = X.select_dtypes(include='number').columns.tolist()
    cats = [c for c in X.columns if c not in nums]
    return ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                              ('scaler', StandardScaler())]), nums),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='缺失')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cats)
    ], remainder='drop', verbose_feature_names_out=True)


def candidates(X):
    def wrap(model):
        return Pipeline([('preprocess', preprocessor(X)), ('model', model)])
    rf = wrap(RandomForestRegressor(n_estimators=120, max_depth=3, min_samples_leaf=3,
                                   max_features=0.8, random_state=SEED, n_jobs=1))
    xgb = wrap(XGBRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                           min_child_weight=3, reg_lambda=10, reg_alpha=0.1,
                           objective='reg:squarederror', tree_method='hist',
                           random_state=SEED, n_jobs=1))
    glm = wrap(GammaRegressor(alpha=1, max_iter=3000, tol=1e-7))
    # Each base learner owns its preprocessing, including Stacking's inner OOF fits.
    stacking = StackingRegressor(
        estimators=[('rf', clone(rf)), ('xgb', clone(xgb)), ('glm', clone(glm))],
        final_estimator=Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=10))]),
        cv=KFold(n_splits=3, shuffle=True, random_state=SEED + 2),
        passthrough=False, n_jobs=1)
    return {
        'RandomForest': (rf, {'model__max_depth': [2, 4], 'model__min_samples_leaf': [2, 4]}),
        'XGBoost': (xgb, {'model__max_depth': [1, 2], 'model__learning_rate': [0.03, 0.08]}),
        'GLM_Gamma': (glm, {'model__alpha': [0.1, 1, 10]}),
        'Stacking': (stacking, {'final_estimator__ridge__alpha': [1, 10, 100]})
    }


def search(estimator, grid):
    return GridSearchCV(estimator, grid, scoring='neg_mean_absolute_error',
                        cv=KFold(n_splits=3, shuffle=True, random_state=SEED + 1),
                        n_jobs=1, refit=True, error_score='raise')


def metrics(y, pred):
    if not np.isfinite(pred).all():
        raise ValueError('预测结果包含无效数值')
    return {'MAE': float(mean_absolute_error(y, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'MAPE_percent': float(100 * mean_absolute_percentage_error(y, pred)),
            'R2': float(r2_score(y, pred))}


def write_report(output, summary, predictions, audit):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, (scene, title) in zip(axes, SCENARIOS.items()):
        s = summary[summary.scenario == scene].sort_values('MAE')
        ax.barh(s['model'], s['MAE'], color=['#1c6e8c' if m != 'MeanBaseline' else '#b4bcc6' for m in s['model']])
        ax.invert_yaxis()
        ax.set(title=title, xlabel='外层折外 MAE（假定元/m²）')
        ax.grid(axis='x', alpha=0.2)
    fig.savefig(output / 'model_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for ax, (scene, title) in zip(axes, SCENARIOS.items()):
        best = summary[(summary.scenario == scene) & (summary.model != 'MeanBaseline')].sort_values('MAE').iloc[0]['model']
        p = predictions[(predictions.scenario == scene) & (predictions.model == best)]
        ax.scatter(p.actual, p.predicted, color='#1c6e8c')
        low, high = min(p.actual.min(), p.predicted.min()), max(p.actual.max(), p.predicted.max())
        ax.plot([low, high], [low, high], '--', color='#777')
        ax.set(title=f'{title}：{best}', xlabel='实际单方造价', ylabel='折外预测')
    fig.savefig(output / 'oof_predictions.png', dpi=160)
    plt.close(fig)
    table = summary.rename(columns={'scenario': '场景', 'model': '模型', 'MAPE_percent': 'MAPE (%)'}).copy()
    table['场景'] = table['场景'].map(SCENARIOS)
    text = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>单方造价模型对比</title>
    <style>body{font:16px/1.7 "Microsoft YaHei",sans-serif;max-width:1100px;margin:36px auto;padding:0 24px;color:#243447}h1,h2{color:#17435a}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce3ea;text-align:right}th{background:#edf4f7}img{width:100%}.note{background:#edf4f7;padding:16px;border-radius:8px}code{background:#eee;padding:2px 4px}</style>
    <h1>单方造价模型对比</h1>
    <p>24 个有标签项目，两种预测场景，四种回归模型及均值基准。主指标 MAE 越低越好，R² 越高越好。</p>
    <p class="note">外层 6 折、每折 4 个验证样本；内层 3 折按 MAE 选参数。以下指标汇总全部 24 个折外预测，每个项目只作一次外层验证。没有额外独立测试集，当前结果用于探索性比较，模型选择后的部署误差仍需新增项目检验。</p>
    <h2>验证结果</h2>'''
    text += table.to_html(index=False, border=0, float_format=lambda n: f'{n:.3f}')
    text += '<img src="model_comparison.png" alt="MAE 对比"><img src="oof_predictions.png" alt="折外预测对比">'
    text += '''<h2>数据与方法</h2><ul>
    <li>早期场景：18 个面积、层数、高度、是否包含工程、结构和基础形式字段。详细场景：另加 25 个分项价格，共 43 个 X。</li>
    <li>项目名称仅作标识，不进入模型。原表结构、基础形式等已经是 0/1 独热编码，本次数据没有额外文字类别 X；管道支持新增文字类别的独热编码和未知类别处理。</li>
    <li>数值缺失值按训练折中位数填补，所有数值 X 使用训练折均值和标准差做 Z-score 标准化，含现有 0/1 字段。y 保留原始单位。</li>
    <li>GLM 使用 Gamma 分布、log 链接及 L2 正则化，预测值为正。Stacking 用 RF、XGBoost、GLM 的内部折外预测训练标准化＋Ridge 元模型，基学习器参数预先固定，元模型正则强度在内层选取。</li>
    <li>没有删除异常价格、没有合成样本；经济指标中的零保持原值。单方造价单位假定为元/m²，原表该列未标单位。</li>
    <li>详细模型要求预测时分项价格确实可得。同一项目重复记录、同源项目关联和未来时间漂移需更多元数据才能评估。</li>
    <li>表1采用两行表头重建列名；最后一条只有项目名称、没有 y 的记录排除。Sheet1 是不完整草稿，Sheet2 是 8 条住宅数据，字段不同，均未混入训练。</li>
    </ul><h2>数据检查</h2>'''
    text += '<pre>' + html.escape(json.dumps({'excluded_without_target': audit['excluded_without_target'], 'area_issues_preserved': audit['area_issues_preserved']}, ensure_ascii=False, indent=2)) + '</pre>'
    text += '<p>面积不一致处保留原值，不自动更改。模型已用全部 24 个有效样本重训并保存，使用方法见项目 README。</p></html>'
    (output / 'report.html').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / 'models').mkdir(exist_ok=True)
    data, features, audit = load_training_data(args.input)
    if len(data) < 18:
        raise ValueError('当前嵌套交叉验证方案至少需要 18 个有效样本')
    y = data[TARGET].to_numpy(dtype=float)
    data.to_csv(output / 'cleaned_data.csv', index=False, encoding='utf-8-sig')
    (output / 'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    splits = list(KFold(n_splits=6, shuffle=True, random_state=SEED).split(data))
    all_metrics, all_predictions, fold_log = [], [], []
    manifest = {'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'seed': SEED, 'outer_cv': 6, 'inner_cv': 3, 'stacking_cv': 3,
                'target': TARGET, 'target_unit': audit['target_unit'], 'sample_count': len(data),
                'glm': 'Gamma/log', 'selection_metric': 'outer OOF MAE', 'scenarios': {}}
    for scene, cols in features.items():
        print(f'开始 {scene}: {len(cols)} 个特征', flush=True)
        X = data[cols]
        models = candidates(X)
        oof = {m: np.full(len(y), np.nan) for m in [*models, 'MeanBaseline']}
        fold_ids = np.zeros(len(y), dtype=int)
        for fold, (tr, va) in enumerate(splits, 1):
            fold_ids[va] = fold
            oof['MeanBaseline'][va] = DummyRegressor(strategy='mean').fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
            for name, (estimator, grid) in models.items():
                fit = search(clone(estimator), grid).fit(X.iloc[tr], y[tr])
                pred = fit.predict(X.iloc[va])
                oof[name][va] = pred
                fold_log.append({'scenario': scene, 'model': name, 'fold': fold,
                                 'validation_excel_rows': data.iloc[va]['Excel行号'].astype(int).tolist(),
                                 'best_params': fit.best_params_, 'inner_MAE': float(-fit.best_score_),
                                 'validation_MAE': float(mean_absolute_error(y[va], pred))})
            print(f'  外层验证 {fold}/6 完成', flush=True)
        trained = {}
        for name, pred in oof.items():
            all_metrics.append({'scenario': scene, 'model': name, **metrics(y, pred)})
            for i in range(len(y)):
                all_predictions.append({'scenario': scene, 'model': name, 'fold': int(fold_ids[i]),
                                        'Excel行号': int(data.loc[i, 'Excel行号']), ID: data.loc[i, ID],
                                        'actual': float(y[i]), 'predicted': float(pred[i]),
                                        'residual': float(y[i] - pred[i]),
                                        'absolute_percent_error': float(100 * abs(y[i] - pred[i]) / y[i])})
            if name == 'MeanBaseline':
                continue
            fit = search(clone(models[name][0]), models[name][1]).fit(X, y)
            bundle = {'model': fit.best_estimator_, 'features': cols, 'numeric_features': X.select_dtypes(include='number').columns.tolist(),
                      'scenario': scene, 'model_name': name, 'target': TARGET, 'seed': SEED,
                      'best_params': fit.best_params_, 'training_feature_ranges': {
                          c: [float(X[c].min()), float(X[c].max())] for c in X.select_dtypes(include='number').columns}}
            filename = f'{scene}_{name}.joblib'
            joblib.dump(bundle, output / 'models' / filename)
            reloaded = joblib.load(output / 'models' / filename)
            np.testing.assert_allclose(fit.predict(X), reloaded['model'].predict(X), rtol=1e-9)
            trained[name] = {'file': f'models/{filename}', 'params': fit.best_params_}
        scene_scores = [r for r in all_metrics if r['scenario'] == scene and r['model'] != 'MeanBaseline']
        best = min(scene_scores, key=lambda r: r['MAE'])['model']
        manifest['scenarios'][scene] = {'label': SCENARIOS[scene], 'features': cols,
                                       'models': trained, 'recommended_model': best}
        template = data[[ID] + cols].iloc[:1].copy()
        template.iloc[0, 0] = '示例项目_请替换全部特征值'
        template.to_csv(output / f'prediction_template_{scene}.csv', index=False, encoding='utf-8-sig')
        print(f'{scene} 完成，折外 MAE 最低：{best}', flush=True)
    summary = pd.DataFrame(all_metrics).sort_values(['scenario', 'MAE'])
    predictions = pd.DataFrame(all_predictions)
    summary.to_csv(output / 'metrics.csv', index=False, encoding='utf-8-sig')
    predictions.to_csv(output / 'oof_predictions.csv', index=False, encoding='utf-8-sig')
    (output / 'cv_details.json').write_text(json.dumps(fold_log, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(output, summary, predictions, audit)
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=1):
        main()
