# 参考文献与引用范围核验报告

核验日期：2026-09-30。方式：AI 辅助原始来源核验；所有记录的 `human_verified=false`。

当前正文和补充材料引用的 **75 条文献均已纳入核验**，已记录的 **23 处文字、归因和书目信息修正均可在当前源文件找到**。未发现缺失引用键或重复 BibTeX 键。本报告不代表独立人工审阅，也不代表每篇出版全文均已取得。

## 数量和机械完整性

| 项目 | 数量/结果 |
|---|---|
| 原始审查 | 75 条 |
| 新增已核验来源 | 1 条：NATA 2018 |
| 审计记录合计 | 76 条 |
| 当前正文引用 | 71 条 |
| 当前补充材料引用 | 19 条，其中 4 条为正文之外新增 |
| 当前正文与补充材料去重引用 | 75 条 |
| 当前 BibTeX | 90 个条目、90 个唯一键 |
| 缺失引用键 / 重复键 / 找不到的输入文件 | 0 / 0 / 0 |
| 当前未引用 BibTeX 条目 | 15 条；其中 Melo 已核验，其余原先未引的 14 条不在此次审查范围 |

原始 75 条全部保留在审计记录中。`melo2022open` 的特定 run 级拆分推荐未能确认，因此正文已撤下该引用；新加入 `nata2018gap`。所以“76 条审计记录”与“当前 75 条引用”并不矛盾。正文之外的 4 条为：`abboush2026sensor`, `chen2014cognitive`, `ricker1995optimal`, `vieira2026realistic`。

机械检查从 `main.tex`、`supplement.tex` 递归读取实际的 `input/include` 文件，剔除注释后枚举引用键；它不执行任意 TeX 宏，也不替代 PDF 编译及视觉检查。逐文件 SHA-256 及当前引用位置见配套 JSON。

## 已完成的纠正

1. 区分既有论文的任务微调与本稿无需任务训练的服务设定。
2. FaultExplainer 对照表补回有/无根因列表的输入条件变化。
3. 删除未核实的 Melo run 级拆分推荐归因。
4. 明确 McNemar 的样本内配对与案例对之间的独立性。
5. Chen 变量身份与 Ricker 单位表分别归因，删除完整实现单位审计的暗示。
6. 删除多数表格基准缺少物理约束的未经支持比较。
7. 区分正确重排的不变性测试与映射错误注入。
8. 将中间位置退化限定到已研究的长上下文任务。
9. 限定自一致性投票增益的任务范围。
10. 补回 LIFT 的监督微调前提。
11. 取消 Tan 文献对本稿六基线呈现选择的不当背书。
12. 将 C-MAPSS 归一化写成已有预处理先例。
13. 将交叉依赖表述改为适当建模/重采样原则。
14. 选择性预测引用仅连接覆盖率和接受错误率。
15. 位置注意力改为关联及干预证据，避免唯一因果归因。
16. 删除覆盖所有不确定性文献的黑箱限定。
17. 区分 Jin 的立场/示例与 Merrill 的零样本直接评估。
18. 收窄 ISO 实验室类比，并增加已读取的 NATA 直接来源。
19. 删除重复的 kassis2026skills BibTeX 键。
20. 按官方会议记录补全 Gruver 页码、DOI 和 URL。
21. 增加 NATA 2018 指南及条款页码。
22. 正文把两份 FD-LLM 报告明确归入同一研究线。
23. 补充材料注明 FD-LLM 两份报告不计作独立复现。

全部 23 项已对照当前文件确认。它们调整引用与叙述范围，没有修改科研结果数值。详细修改日志为 `corrections_applied.json`；JSON 中保留编号对应关系。

## 仍需保留的证据边界

- “已核验”指按可取得的原始来源检查元数据和相关引用命题，不等于所有最终出版全文都已通读。不能取得全文的条目保留摘要、元数据、出版商局部预览或作者版本的明确标记。
- ISO/IEC 17025 的正版全文未读取。已直接查阅 [NATA 官方 April 2018 指南](https://nata.com.au/files/2021/05/17025-2017-Gap-analysis.pdf)第 23 页、7.7.1 条；它支持在适当情形使用参考或质控材料监测结果有效性的有限类比，不证明同批 LLM 检查必然有效，也不构成标准合规认证。
- Chen 的表支持 TEP 变量身份，Ricker 作者配套表 3 和表 4 支持相应变量及单位。本轮未逐字段核对本稿序列化实现；不能把来源检查写成完整单位审计。
- 两份 FD-LLM 报告有相同框架名称、主要作者及方法谱系，已按同一研究线处理。尚无明确官方元数据证明其正式预印本—最终出版版本关系，也未核定全部实验重用比例。
- BatchPrompt 和 Kiangala 已讨论投票/验证收益的条件限制。本稿不能把一致不代表正确、增加投票未必改善这类一般认识当作首次提出。
- 数据集出处与特征/划分设计分别归因。数据文档核验不等于重新分析全部原始测量数据，软件原始论文也不证明当前环境和所有实现版本。

访问层级统计（共 76 条；详细原始访问说明保留在 JSON）：

| 访问类别 | 条目数 |
|---|---|
| 作者稿/预印本相关全文段落 | 9 |
| 摘要、元数据或出版商局部预览 | 10 |
| 相关全文段落/原文扫描（不表示逐句通读） | 51 |
| 官方数据文档（论文全文状态另列） | 4 |
| 作者配套原始操作表 | 1 |
| 摘要及文字编码受限的预印本 | 1 |

上述“相关全文段落”也包括正式会议稿、官方 arXiv 和作者/高校托管的原始文章扫描；具体版本由逐条记录限定，不把作者稿冒充最终出版社版本。

## 逐条结果与可追溯来源

表中给出便于阅读的首要来源链接。每条的全部原始链接、段落/页码定位、原审查判断、已应用纠正及剩余限制见 `reference_audit.json`。原审查判断针对修订前措辞；当前结论由修正编号和“当前结论”共同确定。

| # | 引用键 | 访问层级 | 当前结论 | 原始来源 |
|---|---|---|---|---|
| 1 | `lei2020applications` | 作者稿/预印本相关全文段落 | 期刊元数据与作者接受稿一致；不是最终分页版全文核验。 | [官方/作者来源](https://bura.brunel.ac.uk/handle/2438/20040) |
| 2 | `zhao2019deep` | 作者稿/预印本相关全文段落 | 期刊元数据正确；方法背景由较早作者稿相关段落支持，最终全文未取得。 | [官方/作者来源](https://arxiv.org/abs/1612.07640) |
| 3 | `saxena2008damage` | 摘要、元数据或出版商局部预览 | 仅摘要与元数据支持仿真来源；具体文件以单独 NASA 数据源为准。 | [官方/作者来源](https://ntrs.nasa.gov/citations/20090029214) |
| 4 | `downs1993plant` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://users.abo.fi/khaggblo/RS/Downs.pdf) |
| 5 | `rieth2017additional` | 官方数据文档（论文全文状态另列） | 官方 DataCite 沉积说明支持文件设计；仓库页面受挑战，未重审原始数据。 | [官方/作者来源](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1) |
| 6 | `jin2024time` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://proceedings.iclr.cc/paper_files/paper/2024/hash/680b2a8135b9c71278a09cafb605869e-Abstract-Conference.html) |
| 7 | `liu2024large` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://www.ijcai.org/proceedings/2024/921) |
| 8 | `gruver2023large` | 相关全文段落/原文扫描（不表示逐句通读） | 已按官方 NeurIPS 记录补全页码、DOI 和原始 URL。 修正 20。 | [官方/作者来源](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3eb7ca52e8207697361b2c0fb3926511-Abstract-Conference.html) |
| 9 | `qaid2026explainable` | 摘要、元数据或出版商局部预览 | 已改正无需任务训练的错误归因，与 2024 FD-LLM 稿归为同一研究线；仅出版商局部文本。 修正 22,23。 | [官方/作者来源](https://www.sciencedirect.com/science/article/pii/S0952197625031628) |
| 10 | `zheng2024empirical` | 摘要、元数据或出版商局部预览 | 已把文献中的任务微调与本研究的无需任务训练服务分开；出版商全文未取得。 修正 1。 | [官方/作者来源](https://www.sciencedirect.com/science/article/pii/S095183202400454X) |
| 11 | `constantinides2025failuresensoriq` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://proceedings.nips.cc/paper_files/paper/2025/hash/b7ba530076dee3b4f19847d8dce983da-Abstract-Datasets_and_Benchmarks_Track.html) |
| 12 | `lin2025finetuned` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://aclanthology.org/2025.findings-emnlp.1126/) |
| 13 | `vieira2026realistic` | 作者稿/预印本相关全文段落 | 作者稿支持物理轴承划分原则；应用到其他源单位是本稿设计。 | [官方/作者来源](https://arxiv.org/abs/2509.22267) |
| 14 | `saxena2008dataset` | 官方数据文档（论文全文状态另列） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| 15 | `sensorbench2025` | 摘要、元数据或出版商局部预览 | 摘要及官方项目材料支持代码式传感处理范围；全文未取得。 | [官方/作者来源](https://hotmobile.org/2025/index.php?id=program) |
| 16 | `chen2014cognitive` | 相关全文段落/原文扫描（不表示逐句通读） | 已限定支持变量身份；该文表 1/2 不提供单位列。 | [官方/作者来源](https://petertino.github.io/web/PAPERS/CCE_14.pdf) |
| 17 | `ricker1995optimal` | 作者配套原始操作表 | 作者配套表 3 有 41 个测量变量单位，表 4 前 11 个操纵变量用于本稿；未逐字段核对实现。 修正 5。 | [官方/作者来源](https://depts.washington.edu/control/LARRY/TE/download.html) |
| 18 | `abboush2026sensor` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://arxiv.org/abs/2608.05921) |
| 19 | `kassis2026skills` | 相关全文段落/原文扫描（不表示逐句通读） | 官方 arXiv v2 元数据一致，已删重复键；只能支持工具库归属，不能认证论文正确。 修正 19。 | [官方/作者来源](https://arxiv.org/abs/2609.00065) |
| 20 | `kiangala2026verification` | 相关全文段落/原文扫描（不表示逐句通读） | 一致不代表正确、验证收益有条件已有先例；不能据此主张一般理论首次。 | [官方/作者来源](https://link.springer.com/article/10.1007/s00170-026-19043-z) |
| 21 | `qaid2024fdllm` | 相关全文段落/原文扫描（不表示逐句通读） | 任务微调表述受支持；与相关期刊稿为同一研究线，正式版本对应关系仍未证实。 修正 22,23。 | [官方/作者来源](https://arxiv.org/abs/2412.01218) |
| 22 | `helwig2015hydraulicdata` | 官方数据文档（论文全文状态另列） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) |
| 23 | `hegselmann2023tabllm` | 相关全文段落/原文扫描（不表示逐句通读） | 已删除工业约束在多数表格基准中缺失的未经支持比较。 修正 6。 | [官方/作者来源](https://proceedings.mlr.press/v206/hegselmann23a.html) |
| 24 | `liu2026tabularrobustness` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://academic.oup.com/pnasnexus/article/doi/10.1093/pnasnexus/pgag197/8699520) |
| 25 | `cheng2023batch` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://aclanthology.org/2023.emnlp-industry.74/) |
| 26 | `lin2024batchprompt` | 相关全文段落/原文扫描（不表示逐句通读） | 较低基础性能下投票累积错误、增加轮数未必改善已有讨论；保留新颖性边界。 | [官方/作者来源](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5d8c01de2dc698c54201c1c7d0b86974-Abstract-Conference.html) |
| 27 | `fan2024batcher` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/ICDE24-ER.pdf) |
| 28 | `sclar2024quantifying` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6c0e99d736da621403018ca7b32b1a4d-Abstract-Conference.html) |
| 29 | `mizrahi2024state` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://aclanthology.org/2024.tacl-1.52/) |
| 30 | `lu2022fantastically` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://aclanthology.org/2022.acl-long.556/) |
| 31 | `zhao2021calibrate` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://proceedings.mlr.press/v139/zhao21c.html) |
| 32 | `ribeiro2020beyond` | 相关全文段落/原文扫描（不表示逐句通读） | 已分清正确成对重排的不变性测试与未指定方向的映射错误注入。 修正 7。 | [官方/作者来源](https://aclanthology.org/2020.acl-main.442/) |
| 33 | `zheng2024notrobust` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://proceedings.iclr.cc/paper_files/paper/2024/hash/54dd9e0cff6d9214e20d97eb2a3bae49-Abstract-Conference.html) |
| 34 | `pezeshkpour2024order` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://aclanthology.org/2024.findings-naacl.130/) |
| 35 | `liu2024lost` | 相关全文段落/原文扫描（不表示逐句通读） | 已限定长上下文问答/键值检索和并非恒定成立的中间位置退化。 修正 8。 | [官方/作者来源](https://aclanthology.org/2024.tacl-1.9/) |
| 36 | `hsieh2024found` | 相关全文段落/原文扫描（不表示逐句通读） | 已改为位置注意力关联与校准干预，不把它写成唯一因果解释。 修正 15。 | [官方/作者来源](https://aclanthology.org/2024.findings-acl.890/) |
| 37 | `shi2023distracted` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://proceedings.mlr.press/v202/shi23a.html) |
| 38 | `wang2024fair` | 相关全文段落/原文扫描（不表示逐句通读） | 支持所研究 LLM 评判任务的位置偏差，不自动证明工业诊断同效应。 | [官方/作者来源](https://aclanthology.org/2024.acl-long.511/) |
| 39 | `ouyang2025nondeterminism` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://kclpure.kcl.ac.uk/ws/files/365277933/An_Empirical_Study_of_the_Non-Determinism_of_ChatGPT_in_Code_Generation_Version_of_Record.pdf) |
| 40 | `atil2025nondeterminism` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://aclanthology.org/2025.eval4nlp-1.12/) |
| 41 | `wang2023selfconsistency` | 作者稿/预印本相关全文段落 | 已限定算术/常识任务中的答案投票收益；不作普遍增益保证。 修正 9。 | [官方/作者来源](https://openreview.net/forum?id=1PL1NIMMrw) |
| 42 | `kuhn2023semantic` | 相关全文段落/原文扫描（不表示逐句通读） | 已删除覆盖全部共引方法的黑箱限定；该方法使用生成概率。 修正 16。 | [官方/作者来源](https://openreview.net/forum?id=VD-AYtP0dve) |
| 43 | `xiong2024uncertainty` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6733cf15e10e2cd1d59af033c3bb8507-Abstract-Conference.html) |
| 44 | `manakul2023selfcheckgpt` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://aclanthology.org/2023.emnlp-main.557/) |
| 45 | `sui2024table` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://arxiv.org/html/2305.13062v5) |
| 46 | `fang2024tabular` | 作者稿/预印本相关全文段落 | 期刊元数据配合作者稿支持综述层面的序列化讨论。 | [官方/作者来源](https://www.jmlr.org/tmlr/papers/) |
| 47 | `dinh2022lift` | 相关全文段落/原文扫描（不表示逐句通读） | 已补回以特征和标签文本进行监督微调这一必要条件。 修正 10。 | [官方/作者来源](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4ce7fe1d2730f53cb3857032952cd1b8-Abstract-Conference.html) |
| 48 | `borisov2023great` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://openreview.net/forum?id=cEygmQNOeI) |
| 49 | `li2024chatgptphm` | 作者稿/预印本相关全文段落 | 综述支持基础模型方向；已与其他论文的任务微调/本稿操作选择分开。 修正 1。 | [官方/作者来源](https://arxiv.org/abs/2305.06472) |
| 50 | `tao2025llmbearing` | 作者稿/预印本相关全文段落 | 期刊元数据与作者预印本方法一致；最终出版全文未取得。 | [官方/作者来源](https://arxiv.org/abs/2411.02718) |
| 51 | `tao2025phmlm` | 作者稿/预印本相关全文段落 | 路线图范围获支持；全文证据来自作者稿，期刊全文未取得。 | [官方/作者来源](https://www.sciencedirect.com/science/article/pii/S088832702500384X) |
| 52 | `khan2025faultexplainer` | 作者稿/预印本相关全文段落 | 已补充有/无根因列表的输入条件对照；作者预印本与期刊元数据分开核实。 修正 2。 | [官方/作者来源](https://arxiv.org/abs/2412.14492) |
| 53 | `ren2026phmgpt` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://www.engineering.org.cn/engi/EN/10.1016/j.eng.2025.11.001) |
| 54 | `chen2025largemodels` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://ojs.istp-press.com/dmd/article/view/832) |
| 55 | `jin2024position` | 相关全文段落/原文扫描（不表示逐句通读） | 已与 Merrill 的直接基准分开，限定为立场论文中的示例及解释局限。 修正 17。 | [官方/作者来源](https://proceedings.mlr.press/v235/jin24i.html) |
| 56 | `tan2024useful` | 相关全文段落/原文扫描（不表示逐句通读） | 保留其时间序列消融依据，删除其不能支持的本稿六基线选择依据。 修正 11。 | [官方/作者来源](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html) |
| 57 | `merrill2024struggle` | 相关全文段落/原文扫描（不表示逐句通读） | 已明确所评估模型的零样本时间序列推理范围。 修正 17。 | [官方/作者来源](https://aclanthology.org/2024.findings-emnlp.201/) |
| 58 | `melo2022open` | 摘要、元数据或出版商局部预览 | 未验证其提出 run 级拆分建议；该引用已从当前稿删除，保留原审计记录。 修正 3。 | [官方/作者来源](https://www.sciencedirect.com/science/article/pii/S0098135422003003) |
| 59 | `ramasso2014benchmarking` | 相关全文段落/原文扫描（不表示逐句通读） | 已改为文中回顾过的预处理先例，不再声称 FD002 的普遍标准做法。 修正 12。 | [官方/作者来源](https://papers.phmsociety.org/index.php/ijphm/article/view/2236) |
| 60 | `bates2015lme4` | 相关全文段落/原文扫描（不表示逐句通读） | 已改为交叉随机效应等可选方法；删除未有文献依据的领域普遍性判断。 | [官方/作者来源](https://www.jstatsoft.org/article/view/v067i01) |
| 61 | `dietterich1998approximate` | 摘要及文字编码受限的预印本 | 已明确 McNemar 保留样本内配对，独立性针对不同测试案例对。 修正 4。 | [官方/作者来源](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/dietterich1998.pdf) |
| 62 | `baayen2008mixed` | 相关全文段落/原文扫描（不表示逐句通读） | 已改为处理交叉依赖的原则，不声称方法唯一或所有重采样等价。 修正 13。 | [官方/作者来源](https://www.sfs.uni-tuebingen.de/~hbaayen/publications/baayenDavidsonBates.pdf) |
| 63 | `helwig2015condition` | 官方数据文档（论文全文状态另列） | IEEE 论文全文未取得；具体周期/传感器事实由官方 UCI 配套数据说明支持。 | [官方/作者来源](https://ieeexplore.ieee.org/document/7151267) |
| 64 | `mcnemar1947note` | 摘要、元数据或出版商局部预览 | 支持配对比例检验的历史归属；不是对本稿精确实现的验证。 | [官方/作者来源](https://link.springer.com/article/10.1007/BF02295996) |
| 65 | `holm1979simple` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://www.jstor.org/stable/4615733) |
| 66 | `owen2007pigeonhole` | 相关全文段落/原文扫描（不表示逐句通读） | 已限制为在假设下的可用重采样方法；本稿适配不能自动继承总体覆盖率保证。 | [官方/作者来源](https://arxiv.org/pdf/0712.1111) |
| 67 | `fleiss1971measuring` | 摘要、元数据或出版商局部预览 | 支持多评者名义型一致性指标；不证明预测正确或调用独立。 | [官方/作者来源](https://doi.org/10.1037/h0031619) |
| 68 | `wilson1927probable` | 相关全文段落/原文扫描（不表示逐句通读） | 原始扫描第 209–210 页区间推导已核；不表示历史全文逐句审读。 | [官方/作者来源](https://www.jstor.org/stable/2276774) |
| 69 | `breiman2001random` | 摘要、元数据或出版商局部预览 | 原始摘要与元数据足以支持随机森林算法归属；未读取整篇全文。 | [官方/作者来源](https://link.springer.com/article/10.1023/A:1010933404324) |
| 70 | `pedregosa2011scikit` | 摘要、元数据或出版商局部预览 | 支持软件归属；具体当前算法版本需依本稿环境记录。 | [官方/作者来源](https://jmlr.org/papers/v12/pedregosa11a.html) |
| 71 | `geifman2017selective` | 相关全文段落/原文扫描（不表示逐句通读） | 引用已紧跟覆盖率/接受错误率；本稿产出与调用预算另行定义。 修正 14。 | [官方/作者来源](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) |
| 72 | `oleson2011programmatic` | 相关全文段落/原文扫描（不表示逐句通读） | 所引背景或方法在已检查范围内获得支持；不将该来源视为本研究结果的独立验证。 | [官方/作者来源](https://cdn.aaai.org/ocs/ws/ws0800/3995-16701-1-PB.pdf) |
| 73 | `iso17025` | 摘要、元数据或出版商局部预览 | 正版标准全文未读取；类比已收窄，由新增官方 NATA 文件佐证，不能宣称合规认证。 修正 18。 | [官方/作者来源](https://www.iso.org/standard/66912.html) |
| 74 | `chen2023aspire` | 相关全文段落/原文扫描（不表示逐句通读） | 适配训练后选择性预测表述受支持；不得写成纯免训练 API 包装。 | [官方/作者来源](https://aclanthology.org/2023.findings-emnlp.345/) |
| 75 | `lessmeier2016benchmark` | 相关全文段落/原文扫描（不表示逐句通读） | 支持加速寿命试验形成的真实损伤台架来源；不是工厂自然运行外部验证。 | [官方/作者来源](https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf) |
| 76 | `nata2018gap` | 相关全文段落/原文扫描（不表示逐句通读） | 官方 PDF 第 23 页 7.7.1 支持在适当情形使用参考/质控材料监测结果有效性；不是同批 LLM 方法的证明。 修正 18,21。 | [官方/作者来源](https://nata.com.au/files/2021/05/17025-2017-Gap-analysis.pdf) |

## 版本和来源快照

FD-LLM：[原始 arXiv 记录](https://arxiv.org/abs/2412.01218)、[相关期刊 DOI 元数据](https://api.crossref.org/works/10.1016%2Fj.engappai.2025.113131)。两者有明确共同研究谱系，但正式版本关系未得到明确链接确认。

Scientific Agent Skills：[官方 arXiv 记录](https://arxiv.org/abs/2609.00065)、[v2 全文](https://arxiv.org/html/2609.00065v2)。复查作者、题名、年份及标识一致；仅去除重复键。该资料不能替代作者对本稿准确性的责任。

公开报告未附外部论文全文或大段逐字引文。网络检索使用公开题名、作者、DOI 和 URL；未发表正文在本地对照。下载来源缓存不属于公开报告交付物。

本报告源文件快照 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `results/industrial_interface_focused_overleaf/main.tex` | `409d16a5eb2084b04e825ff5fb5ae1195ec2c3b1ac85422fe2997fd680eb039a` |
| `results/industrial_interface_focused_overleaf/canary_control_main.tex` | `7512c2c38aeb126ec5a0ccc8c859a69a01f1f5ac6788da0da48113ba2eb6af56` |
| `results/industrial_interface_focused_overleaf/canary_stability_main.tex` | `1531034f790b17aed4fb8fb9eca91bb084ba235c276dc4dfd3b48a08a29062f3` |
| `results/industrial_interface_focused_overleaf/author_biographies.tex` | `e6a849ca99e1735733fdd1c85c7f6842e8b46ca4762f29a512d0fbcad8385dfd` |
| `results/industrial_interface_focused_overleaf/supplement.tex` | `79bb8e87efdb5ee4a62a13e6158118f433ac1483d35c3a842888205971a2ab93` |
| `results/industrial_interface_focused_overleaf/supp_stats_tables.tex` | `11434ca94ebd475d2cb0975d2db1b5302703d4c32ac863ddcc74558fe52253f9` |
| `results/industrial_interface_focused_overleaf/supp_structure_tables.tex` | `26afc6a9b62e3914c2373d7e9cee1ac3720c286d4217a5747f12a72208e26a96` |
| `results/industrial_interface_focused_overleaf/supp_baseline_tables.tex` | `bb4473285b0ec23cd8ef581e2a13ef458caa50be94ef1ae29c75538fd1f121c4` |
| `results/industrial_interface_focused_overleaf/supp_prospective_canary.tex` | `49397a0877317864464524b630192cef8260a885a3db865997ec00c73adf4589` |
| `results/industrial_interface_focused_overleaf/supp_canary_controls.tex` | `6dc91ec305d572c96061660e21dbab54cfefa1e42565ed21e0357b838a2cfb69` |
| `results/industrial_interface_focused_overleaf/supp_canary_controls_table.tex` | `54eb584ebf593379260b3fc122c655da7a98e0d430b88dbd768e45c500459b87` |
| `results/industrial_interface_focused_overleaf/supp_canary_stability.tex` | `83437ce325ddf7e599626fcafc71333e7658d0c1610349da91717ad2fb58ea38` |
| `results/industrial_interface_focused_overleaf/references.bib` | `fa8ff368b77034b25005fd28000589e1d0371b3caab1708f3207e77cd44960cf` |

完整核验数据：[reference_audit.json](reference_audit.json)。修改日志：[corrections_applied.json](corrections_applied.json)。
