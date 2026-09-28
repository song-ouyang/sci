# Claude 当前记录：27 份数组，受控主结果全部 N/A

本目录目前包含 Sonnet 4.6、Opus 5.5、Fable 5.1 三个记录标识，各九份预测。当前审计已覆盖全部 **27** 份：数组结构均有效、23 份当前本地文件链一致、**0 份通过受控主比较验收**。文件链一致并不证明调用身份、实际完整输入、隔离和评分前冻结。

## 当前报告和历史事实

- 当前 `analysis/scores.json`、CSV 和 summary：27 份受控准确率/F1 全部 N/A，不再只列旧 18 份，也不把污染记录算作 100%。
- Sonnet `tep_misnamed_r3` 历史上有 CONTAMINATED 标记且旧评分曾计 100%；后来被手工替换、移除标记。当前原始回复仍未通过完整 JSON 检查，来源链未补齐。
- Sonnet `tep_clean_r2`、`tep_misnamed_r1` 的哈希不一致；`tep_clean_r3` 缺请求且响应为含真实标签的事后说明。这些缺陷保留在审计中，没有事后修好原记录。
- 缺少预测评分前冻结和可核验的原始调用事件；工具读取、分页、续接与原协议不一致。
- Fable metadata 的调用别名为 `opus`，原生身份未核实；保存快照的 TEP 三次配对翻转为 6/70、3/70、0/70，不能支持“全部预测相同”的报告说法。

当前完整审计：[audit/report.md](audit/report.md)。用户提供的九行均值/标准差保存在 `../claude_reported_summary_2026-09-27/`，作为未验收报告进入论文附录 B / 表 14，不替代这里的受控 N/A 结果，也不进入主比较图。

历史原文件、旧评分与旧审计分别保留在 `../claude_extension_audit_archive/2026-09-27/` 和项目 `paper/history/`。这些备份都是事后保留，不能追溯证明实验当时已经冻结。

## 离线复核

```bash
.venv/bin/python src/analysis/score_revision_v3_claude.py --validate-only
```

预期退出码 2，表示验收阻断；不读取答案、不计算分类分数、不更改原预测。省略 `--validate-only` 会更新当前审计与 N/A 派生报告。审计会纳入新增模型目录，并保留缺失分组的失败状态，不悄悄遗漏它们。

## 规范重跑

按 `paper/CLAUDE_EXTENSION_REMEDIATION_2026-09-27.md` 执行。当前 CLI 已安装且本地显示登录，但合成预检的真实调用返回 403 Request not allowed；新目录的正式会话数仍为 0。见 `../revision_v3_claude_extension_v2/BLOCKERS.md`。

主论文已更新，原始 GPT 冻结数值未改动。排除受控评分不等于断定所有标签都错误；预测一致也不能证明内部推理机制。
