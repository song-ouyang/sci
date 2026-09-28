# DeepSeek / Qwen 追加实验结果

完整有效会话：15/18；有效预测：990/1200。
复用 130 个独立来源单位；重复调用不增加独立样本。以下均为第 1 次重复（预设主结果）。

| 模型 | C-MAPSS 准确率 | TEP 原始准确率 | TEP 错配准确率 |
|---|---:|---:|---:|
| deepseek-v4-pro | 66.7% | 62.9% | N/A |
| qwen3.8-max | 53.3% | 42.9% | 20.0% |

## 所有重复

| 模型 | 条件 | 重复 | 状态 | 准确率 | Macro-F1 |
|---|---|---:|---|---:|---:|
| deepseek-v4-pro | cmapss/clean | 1 | valid | 66.7% | 66.5% |
| deepseek-v4-pro | cmapss/clean | 2 | valid | 66.7% | 67.5% |
| deepseek-v4-pro | cmapss/clean | 3 | valid | 61.7% | 61.0% |
| deepseek-v4-pro | tep/clean | 1 | valid | 62.9% | 57.4% |
| deepseek-v4-pro | tep/clean | 2 | valid | 74.3% | 73.3% |
| deepseek-v4-pro | tep/clean | 3 | valid | 34.3% | 27.8% |
| deepseek-v4-pro | tep/misnamed | 1 | invalid_response | N/A | N/A |
| deepseek-v4-pro | tep/misnamed | 2 | valid | 15.7% | 12.1% |
| deepseek-v4-pro | tep/misnamed | 3 | invalid_response | N/A | N/A |
| qwen3.8-max | cmapss/clean | 1 | valid | 53.3% | 50.1% |
| qwen3.8-max | cmapss/clean | 2 | valid | 61.7% | 59.4% |
| qwen3.8-max | cmapss/clean | 3 | valid | 60.0% | 58.9% |
| qwen3.8-max | tep/clean | 1 | valid | 42.9% | 33.5% |
| qwen3.8-max | tep/clean | 2 | valid | 51.4% | 45.6% |
| qwen3.8-max | tep/clean | 3 | valid | 44.3% | 36.8% |
| qwen3.8-max | tep/misnamed | 1 | valid | 20.0% | 12.3% |
| qwen3.8-max | tep/misnamed | 2 | invalid_response | N/A | N/A |
| qwen3.8-max | tep/misnamed | 3 | valid | 37.1% | 31.9% |

失败会话保留原始记录且不重跑；主会话无效时不以其他重复替代。
这是观察既有结果后的探索性追加实验，服务接口与推理参数并不完全等价；两套数据均为仿真数据。
完整混淆矩阵与类别指标见 scores.json；逐条预测见 cases.csv。

本轮是预算修正版 v2。此前 v1 因预算配置不匹配停止：4 个无效会话、2 个操作中断会话、12 个未执行会话。
旧记录完整保存在相邻 domestic_extension 目录；没有替换、删除或混入本轮结果。
