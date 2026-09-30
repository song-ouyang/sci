# Claude v2 当前阻碍：CLI 实际请求被拒绝

2026-09-27 第三次重新检查（~18:30 UTC）。

## 本轮复查（2026-09-27 ~18:30 UTC）

用户通过当前 Claude Code 对话（claude-sonnet-4-6）发起执行请求。

### CLI 调用状态更新

| 调用方式 | 上次状态 | 本次结果 |
|------|------|------|
| `claude auth status` | 已登录 firstParty/claude.ai | 仍然已登录 |
| `claude -p “...”` 直接调用 | 403 Request not allowed | **Credit balance is too low** |
| `claude_v2_preflight.py` 合成预检 | 403 Request not allowed | 仍然 403 Request not allowed |

`subscriptionType: null`（当前账户没有查到订阅类型）。

### 认证与计费说明更正（2026-09-28）

先前“claude -p 必须有 API 积分”“订阅不支持 -p”“充值是唯一路径”的判断没有依据，现撤回。官方支持使用有权限的 Claude 订阅登录 CLI；Console/API 是另一种计费路径。参数 -p 本身不是必须使用 API 余额的证明。

已实查：当前 Codex 环境 `authMethod=claude.ai`、`apiProvider=firstParty`、`apiKeySource=/login managed key`、`subscriptionType=null`。null 只表示此输出未给出订阅类型，不能据此断言无订阅或余额为零。当前进程未发现 ANTHROPIC_* 凭证变量，用户 settings.json 没有 env 块；不能据此排除另一运行环境的差异。

保留上面的 Credit balance is too low 为 Claude 执行者报告的错误，但本目录找到的原生合成预检证据仍为 403。两者不能混作同一次调用或同一个已证实根因。请保存余额报错那次的准确命令、脱敏环境信息、退出码及原始输出。

下一步先在真正能使用 Claude Code 的环境核对 /status 与 claude auth status、认证来源及账号权限。官方说明 ANTHROPIC_API_KEY 在 -p 模式下存在即会被使用，可能与交互模式的选择不同；也要检查 token、apiKeyHelper、配置目录及组织设置，但不打印凭证值、不绕过管理策略、不自动删除配置。

如果用户有可用 Claude Code 订阅，在自己的终端运行 `claude auth login --claudeai`，用对应账号登录，再执行短合成预检。不要使用 `--console`，除非用户明确选择 API 计费。不要用 `--bare` 测试订阅登录，该选项的本机帮助明确排除 OAuth/keychain。登录本身不保证账号/模型访问问题解决。

正式会话仍为 **0/18**。两个目标模型的完整预检通过后才冻结正式协议。若账号确无可用订阅/组织权限，或用户选择 Console，再讨论付费方案；API 充值也不保证能使用旧记录所标注的具体模型 ID。当前不购买、切换计费或启动正式调用。

官方来源（2026-09-28 核对）：
- https://code.claude.com/docs/en/authentication （登录方式及认证优先级）
- https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan （订阅与可选 API 计费）
- https://code.claude.com/docs/en/cli-reference （auth login --console 与订阅的区别）

## 上轮复查与交接（2026-09-27 15:13 UTC）

- 新的短预检：`preflight/20260927T150613Z-b77dda39/assessment.json`，仍为 403；正式会话 0/18。
- 旧数据重新审计：27 份数组格式通过，23 份文件链一致，0 份主比较合格。结果见 `legacy_recheck_20260927.json`。文件链一致不等于完整来源证据。
- 新增 `src/experiments/claude_v2_long_preflight.py`，只提交合成数字，检查整批 ID、首尾标记、工具事件、原生模型/用量/停止事件、完整响应；8 项离线单元测试通过。
- `preflight/long-20260927T151358Z-b90f5672/` 只是离线准备，不曾提交长输入。核对九个旧输入哈希通过，最大为 223,740 UTF-8 字节；合成输入覆盖此字节数。并未证明 token 等价、实际服务长输入通过或正式重试合规。
- 旧 PDF 和已验证复现 ZIP 不变；本轮交接/预检文件尚未重新纳入旧 ZIP。论文表号已为附录 B / 表 18。

## 已实测的事实

- Claude CLI 已安装，版本 **2.1.283**。
- `claude auth status` 返回已登录，认证方式 `claude.ai`，提供方 `firstParty`。
- 项目未配置 Anthropic API Key。本次使用 CLI 自己的既有登录；没有读取或复制登录密钥。
- 已在独立临时目录，用纯合成文本执行一次 `claude-sonnet-4-6` 预检。未提交论文样本、标签或历史结果。
- CLI 启动事件显示 **tools=[]、mcp_servers=[]**，实际参数及输入已留存。
- 实际请求返回 **`Failed to authenticate. API Error: 403 Request not allowed`**；进程退出码 1，结果 `is_error=true`。即使 result subtype 写作 success，也不能将该请求判为成功。
- 错误消息的 model 是 `<synthetic>`，不是已核实的服务端模型身份。

## 因此还没有完成的步骤

完整长度输入与首尾标记校验、成功响应及原生模型身份核验、合成整批 JSON 输出、正式协议冻结、18 个正式会话、正式输出冻结和评分均未完成。

**正式会话数仍为 0。** 本次失败预检单独留存，不纳入 18 个正式会话，不生成假预测或正式冻结清单，也不退回主对话分页读取模式。

## 可复核证据

- `environment_check.json`：已脱敏的 CLI 和认证状态。
- `preflight/cli_help.txt`：实际安装版本的选项说明。
- `preflight/20260927T110854Z-5425a912/submitted_request.json`：准确 CLI 参数和合成输入。
- 同目录 `stdout.jsonl`、`stderr.txt`、`assessment.json`、`artifact_hashes.json`：原始事件、错误、判定和哈希。
- 预检实现：`src/experiments/claude_v2_preflight.py`。

## 下一步

先解决当前 CLI 账户对推理请求的 403 授权拒绝。可在本机终端通过 `claude auth login` 更新登录，并确认该账户获准使用相应服务；重新登录本身不保证消除服务访问策略限制。不要把 token 发到聊天中，也不要绕过账户策略。

授权修复后，可先重做同一合成初步预检：

```bash
.venv/bin/python src/experiments/claude_v2_preflight.py --model claude-sonnet-4-6
```

此命令会新建独立预检记录，绝不覆盖旧结果。初步成功后仍须按整改交接执行最大长度和两个模型的完整预检，才进入正式协议与运行步骤。

参数依据：[官方 CLI 文档](https://code.claude.com/docs/en/cli-reference)及保存的本机 `--help`；本次使用 `--tools ""`，没有猜测 `--tools none`。
