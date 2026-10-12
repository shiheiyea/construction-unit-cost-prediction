# 单方造价预测

输入：`Material/工作簿1.xlsx` 中的“表1”，24 个带 y 的项目。

本次嵌套交叉验证结果：

| 场景 | MAE 最低的模型 | MAE | RMSE | MAPE | R² |
| --- | --- | ---: | ---: | ---: | ---: |
| 项目特征 | Gamma GLM | 675.05 | 840.14 | 22.55% | 0.560 |
| 项目特征＋分项价格 | XGBoost | 712.35 | 857.78 | 21.23% | 0.541 |

同一划分下均值基准的 MAE 为 1196.62。以 MAE 为主指标，当前优先使用项目特征场景的 Gamma GLM；加入分项价格没有改善本次比较的最优 MAE，不能据此推断这些价格在更大数据集上没有预测价值。

两种场景各保存随机森林、XGBoost、Gamma GLM、Stacking 四个模型：

- `early`：18 个项目特征，适用于分项价格尚未知的估算。
- `detailed`：43 个特征，包含 25 个分项价格，预测时必须已知这些价格。

项目名称仅作标识。原表结构、基础形式等已用 0/1 独热编码，不重复编码。预处理支持真正的文字类别字段：训练折内填补缺失、独热编码，并允许预测时出现新类别。全部数值 X 按训练折做 Z-score 标准化，包括原有 0/1 列。y 不标准化，结果维持原始造价单位。原表 y 没标单位，报告暂按元/m²解释。

## 查看结果

打开 `outputs/report.html`，查看两种场景四个模型及均值基准的 MAE、RMSE、MAPE、R²。

- `outputs/metrics.csv`：模型比较，按 MAE 排序。
- `outputs/oof_predictions.csv`：逐项目折外预测和误差。
- `outputs/data_audit.json`：排除记录、缺失与面积差异。
- `outputs/cv_details.json`：每折验证项目、所选参数与误差。
- `outputs/manifest.json`：特征、源数据哈希、模型路径、推荐模型。
- `outputs/models/`：8 个模型文件，均含完整预处理管道，已用 24 个样本重训。

## 预测新项目

复制 `outputs/prediction_template_early.csv` 或 `prediction_template_detailed.csv`，保持列名，替换示例的全部特征值，每行一个项目。模板中的数值来自第一个历史项目，仅示范格式；不能原样当作新项目数据。

在项目根目录用 PowerShell 执行：

```powershell
.\.venv\Scripts\python.exe predict.py --input "你的新项目.csv" --scenario early --output "outputs\新项目预测.csv"
.\.venv\Scripts\python.exe predict.py --input "你的新项目.xlsx" --scenario detailed --model Stacking --output "outputs\详细预测.csv"
```

输入支持 CSV（UTF-8）和 XLSX（第一张表，单行表头）。默认 `--model best` 使用该场景外层 MAE 最低的算法；也可指定 `RandomForest`、`XGBoost`、`GLM_Gamma` 或 `Stacking`。缺列或非数字内容会报错，存在但为空的数值会用训练中位数填补。输出保留输入列，添加预测值、场景、模型和输入检查；超出训练数值范围会标记。RF、XGBoost、Ridge 元模型未强制正值，非正预测会标记且保留原值。

## 重新训练与复现

现有 `.venv` 已安装依赖，可直接运行：

```powershell
.\.venv\Scripts\python.exe train.py
```

验证保存模型、折外指标和预处理行为：`.\.venv\Scripts\python.exe verify.py`。

重训会更新 `outputs` 内的模型和报告，请先另存需要保留的结果。迁移环境时建议 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe train.py
```

训练随机种子为 20261012。外层 6 折（每折验证 4 个项目），内层 3 折按 MAE 选参数。两个场景、所有模型的外层划分相同。所有编码、缺失填补和标准化都在训练折内拟合。Stacking 的每个基学习器有独立预处理管道，内部 3 折生成折外预测，使用标准化后的基学习器预测训练 Ridge 元模型。Stacking 的基学习器采用事先固定的保守参数，内层仅选择元模型的正则强度，避免将已见过验证样本的预处理器带入 Stacking。

随机森林搜索深度和叶节点最小样本数；XGBoost 搜索深度和学习率；GLM 搜索 L2 正则强度。Gamma GLM 使用 log 链接，对正值 y 建模。均值基准仅使用每个外层训练折的 y 均值。

## 结果边界

当前只有 24 条样本，详细场景特征数多于样本数。结果是嵌套交叉验证的探索性比较，没有独立测试集；推荐算法也是按这些外层结果选择，不能把其最优误差当作未来部署误差保证。没有年份、独立项目组等元数据，本次无法验证未来时间泛化或同源项目关联。补充真实新项目后应另做独立验证，不因训练集拟合好就宣称准确。

最后一条只有项目名、无 y 的记录不参与训练。经济指标中的零保留，不推断为缺失；面积不一致处保留原值。其余两个工作表结构不同，不合并训练。

实现参考：[预处理和泄漏](https://scikit-learn.org/stable/common_pitfalls.html)、[GammaRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html)、[StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)。
