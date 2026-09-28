# 重复调用与条件差异：事后描述性审计

这是既有冻结输出的离线再分析，不是新实验、预注册指标或独立人类同行评审。未读取逐例真实标签。

“观察到的重复稳定差异”仅表示：指定的全部重复中，两条件各自标签完全一致且两侧不同。它不是总体稳定性、显著性或因果证据。

TEP的R编号不是匹配的执行时间块；条件与调用时间仍可能混淆。这些稳定计数不隔离输入效应，也不是因果效应的下界。没有观察到稳定差异不等于没有条件效应。

每个条件内部列出全部重复对；条件之间列出所有交叉重复对。配对复用同一组案例和调用，不能当作独立样本、用于扩大 n，或相减形成净效应。

| 数据 / 模型 | 条件对 | 案例 | 重复/侧 | 稳定不同 | 稳定相同 | 任一不稳定 | 重复缺失 |
|---|---|---:|---:|---:|---:|---:|---:|
| tep_three_repeats / gpt-5.6-sol | clean → rotation | 70 | 3 | 20 | 10 | 40 | 0 |
| tep_three_repeats / gpt-5.6-sol | clean → random_1 | 70 | 3 | 0 | 27 | 43 | 0 |
| tep_three_repeats / gpt-5.6-sol | clean → random_2 | 70 | 3 | 0 | 46 | 24 | 0 |
| tep_three_repeats / gpt-5.6-sol | clean → random_3 | 70 | 3 | 0 | 41 | 29 | 0 |
| tep_three_repeats / gpt-5.6-sol | clean → same_unit_swap | 70 | 3 | 0 | 49 | 21 | 0 |
| tep_three_repeats / gpt-5.6-sol | clean → pair_shuffle | 70 | 3 | 0 | 67 | 3 | 0 |
| tep_three_repeats / gpt-5.6-sol | a_front → a_back | 21 | 3 | 0 | 13 | 8 | 0 |
| tep_three_repeats / gpt-5.6-sol | b_front → b_back | 21 | 3 | 0 | 9 | 12 | 0 |
| tep_three_repeats / gpt-5.6-sol | a_front → b_front | 21 | 3 | 0 | 13 | 8 | 0 |
| tep_three_repeats / gpt-5.6-sol | a_back → b_back | 21 | 3 | 0 | 9 | 12 | 0 |
| tep_three_repeats / gpt-6-astra | clean → rotation | 70 | 3 | 0 | 70 | 0 | 0 |
| tep_three_repeats / gpt-6-astra | clean → random_1 | 70 | 3 | 0 | 70 | 0 | 0 |
| tep_three_repeats / gpt-6-astra | clean → random_2 | 70 | 3 | 0 | 70 | 0 | 0 |
| tep_three_repeats / gpt-6-astra | clean → random_3 | 70 | 3 | 0 | 70 | 0 | 0 |
| tep_three_repeats / gpt-6-astra | clean → same_unit_swap | 70 | 3 | 0 | 70 | 0 | 0 |
| tep_three_repeats / gpt-6-astra | clean → pair_shuffle | 70 | 3 | 0 | 70 | 0 | 0 |
| tep_three_repeats / gpt-6-astra | a_front → a_back | 21 | 3 | 0 | 20 | 1 | 0 |
| tep_three_repeats / gpt-6-astra | b_front → b_back | 21 | 3 | 0 | 20 | 1 | 0 |
| tep_three_repeats / gpt-6-astra | a_front → b_front | 21 | 3 | 0 | 20 | 1 | 0 |
| tep_three_repeats / gpt-6-astra | a_back → b_back | 21 | 3 | 1 | 20 | 0 | 0 |
| hydraulic_two_complete_blocks / gpt-5.6-sol | clean → pair_reverse | 18 | 2 | 0 | 0 | 18 | 0 |
| hydraulic_two_complete_blocks / gpt-5.6-sol | clean → descriptor_rotation | 18 | 2 | 0 | 1 | 17 | 0 |
| hydraulic_two_complete_blocks / gpt-5.6-sol | a_front → a_back | 18 | 2 | 0 | 6 | 12 | 0 |
| hydraulic_two_complete_blocks / gpt-5.6-sol | b_front → b_back | 18 | 2 | 0 | 3 | 15 | 0 |
| hydraulic_two_complete_blocks / gpt-5.6-sol | a_front → b_front | 18 | 2 | 0 | 2 | 16 | 0 |
| hydraulic_two_complete_blocks / gpt-5.6-sol | a_back → b_back | 18 | 2 | 0 | 6 | 12 | 0 |
| hydraulic_two_complete_blocks / gpt-5.6-sol | clean → clean_repeat | 18 | 2 | 0 | 1 | 17 | 0 |
| hydraulic_two_complete_blocks / gpt-5.6-sol | a_front → a_front_repeat | 18 | 2 | 0 | 6 | 12 | 0 |

液压结果若纳入，仅单列两个完整执行块（R1/R2）。R3部分完成、R4/R5未执行仍属于原试验缺失；这里不以晚轮补位。clean与clean_repeat是两个已规定的相同输入条件；其同块调用对与跨块调用对分别由pairing_scope字段标记，不能将全交叉配对范围误读为同块duplicate波动。不同数据集及两轮/三轮稳定计数不可横向排名。

输出：contrasts.csv 给出条件级计数和观察范围；within_condition_pairs.csv 与 cross_condition_pairs.csv 保留每一对；case_stability.csv 保留各ID的全部标签及分类；audit.json 记录输入哈希和每个条件的字节一致性核验。条件内配对在不同对照中可重复出现，comparison列标识所属对照。

结论边界：大量一次调用差异可以伴随条件内不稳定；少数指定条件也可能在所有已观察重复中各自保持不同标签。前者限制单次效果解释，后者提供可继续验证的具体案例。二者均不能排除抽样、服务时间或有限重复带来的偶然性。

## 本次观察

Sol的clean/rotation在70例中有20例观察到重复稳定不同、10例稳定相同、40例至少一侧不稳定。其余五个映射对照与四个固定规模上下文对照均未观察到两侧各三轮稳定且不同的案例；这不表示不存在条件差异。
Astra的六个映射对照均无观察到的稳定不同；A-back/B-back在21例中有1例稳定不同、20例稳定相同。另三个上下文对照各有一例至少一侧不稳定。

## 离线重建

```sh
.venv/bin/python src/analysis/audit_repeat_condition_v2.py --include-hydraulic
.venv/bin/python -m unittest tests.test_repeat_condition_v2 -v
```
