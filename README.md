# memory-agent

一个类比人脑的**持续学习 Agent 记忆框架**。纯 Python 标准库实现，零第三方依赖即可运行，可选接入云端 LLM 增强。

> 记忆的本质，是对「有限上下文窗口」的工程绕行。

- **接手 / 二次开发：先读 [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)（上手路径 + 概念词典 + 工作流 + 调试手册）**
- 方案与进度：`Ultimate-GOAL/PLAN.md`（演进方案）、`Ultimate-GOAL/PROJECT-RECORD.md`（项目记录）
- 架构规范：`ARCHITECTURE.md`（分层/职责/依赖规则/未来插入点，由 `tests/unit/test_architecture.py` 守护）
- 路线图：`ROADMAP.md`
- 当前版本：**V1.8.0（记忆认知升级轮）**：P0 止血（助手侧成败词表停用 + 存量清理 + 助手噪声入库率指标）+ P1 排序修复（融合分池保底 + 置信权重纠偏 + dense 置信阈值重标定 + 三路联想种子 + 深搜模式）+ P2 认知升级（图式归纳 rule 层 / 条件上下文 / 信念分级修正 / 只读命令沙箱与原生 list_dir/read_file）；工具轮上限三入口可设置；11 个记忆工具；**598 个测试全绿，十六项评测指标全 100%**

## 类脑映射

| 人脑机制 | 本模块 |
|---|---|
| 前额叶注意力控制 | `attention`：重要性门控，过滤低价值输入（显式声明保底不被误杀） |
| 海马体编码 | `encoding` + `pipeline`：情景/语义/程序三类编码 |
| 情景/语义/程序记忆 | `memory` 三仓储（SQLite + FTS5 中文 2-gram 索引 + 向量）；**分层寿命**——经验类情景半衰 7 天 + 独立归档阈值（LFU 低频先忘、访问续命），普通 30 天；技能可被检索带出（`[技能]` 标签，成功率/使用次数溯源） |
| 工作记忆 | `memory/working`：纯内存会话级 scratchpad（容量 9、salience 淘汰、24h 蒸发），会话结束即忘——瞬时信息的去处 |
| 联想激活 | `retrieval/activation`：实体图两跳扩散（ACT-R 式），关键词与向量都够不着的事实经联想带出（`meta.associated` 标记，只读不写回） |
| 突触可塑性 | `retrieval` 检索后再巩固（强度重算 + SM-2 隐式复习）+ `learning/strength` 可解释强度模型 |
| 睡眠巩固 | `consolidation`：多轮巩固（NREM×3 聚类摘要蒸馏 + REM 跨簇联想——「梦」的工程化），高唤醒记忆优先处理，冲突消解（蒸馏默认挂起待裁） |
| 主动遗忘 | `forgetting`：强度重算 → 归档 → 摘要降级（可溯源）→ 硬删；干扰论遗忘横跨两侧：检索侧提取诱发遗忘（回忆 A 抑制同实体未够到的竞争者）+ 写入侧线索过载（同实体事实过多时最旧降权） |
| 时序记忆 | `valid_from/valid_to` + `superseded_by` 版本链，偏好变更有历史 |
| 元记忆 / 元认知 | `memory/meta` 审计日志 + `reports` 健康报告（含检索空缺）+ `retrieval` 读侧元认知（E7：低置信表面化 `meta.uncertain`、命中溯源 provenance、检索空缺自知） |
| 图式归纳（睡眠期跨事件提炼规律） | `consolidation` 第四阶段 Schema Consolidator：≥2 条跨时间相似情景 → `relation="rule"` 规律（断路器：数量/跨度/LLM 合法性/证据链机械等值；pending 起步、检索命中 3 次转正） |
| 认识论语境（记忆的生效前提） | `SemanticFact.validity_context`（preconditions/environment/expires_if，白名单规范化守门，注入行条件展示；过滤待数据积累） |
| 信念修正（预测误差驱动更新） | 信念分级修正：L1 显式取代 / L2 用户反驳降权（−0.2） / L3 工具结果步进与双次失败翻转（收回待重证可挣回） / L4 纯推演永不改写 |
| 现实淬炼（行动-结果耦合） | `run_command` 只读命令沙箱（白名单 + argvec + 元字符拒绝 + 审计，退出码判 outcome 淬炼经验与技能统计）+ 原生 `list_dir`/`read_file`（路径 confinement，跨平台） |
| 深度回忆（意图展开多路召回） | `retrieval/deep.py`：LLM 把查询展开成 2~3 条线索多路召回合并（`--deep` / `/deep` / 工具参数），线索绝不入库、全程只读、LLM 不可用静默降级快搜 |

## 记忆生命周期

```
感知 → 门控打分（显式声明/绿色通道保底）→ 编码（情景+语义+程序；经验类标记分层 category）
     → 冲突消解（分级：显式与绿色通道取代 / 低置信挂起——环境状态「最新即正确」）
     → 检索（working 置顶 + FTS + 向量混合 + 联想命中追加 + 技能带出 + 置信度表面化[低分标「不确定」]）
     → 再巩固（强度重算，检索/技能带出即续命）→ 遗忘（经验层短半衰快淘汰，摘要降级可溯源）
```

## 快速开始

### 0. 选运行模式

- **纯离线**：什么都不配，规则打分 + 哈希嵌入，开箱即用，评测/测试全程离线
- **接入云端**：项目根目录放 `.env` 配置三通道 LLM（见下），写入更聪明、整理更自动

### 1. 配置 `.env`（可选）

当前在用的双模型分治配置——**主大脑用强模型，记忆梳理用低配模型**：

```ini
# 聊天/打分/抽取（主通道）：OpenRouter 免费 ox-alpha
# 云端客户端强制 IPv4 直连、无视 HTTPS_PROXY 等代理环境变量（走本机 Clash 代理时
# OpenRouter 的 POST 会被节点掐断），任何终端直接运行即可
MEMAGENT_CLOUD_URL=https://openrouter.ai/api/v1
MEMAGENT_CLOUD_API_KEY=sk-or-v1-你的key
MEMAGENT_CLOUD_MODEL=stealth/ox-alpha
MEMAGENT_CLOUD_FALLBACK_MODELS=

# 记忆梳理（维护通道）：SiliconFlow 免费 GLM-4-9B，低配够用且额度宽松
MEMAGENT_MAINT_URL=https://api.siliconflow.cn/v1
MEMAGENT_MAINT_API_KEY=sk-你的key
MEMAGENT_MAINT_MODEL=THUDM/GLM-4-9B-0414

# 工具调用轮数上限（默认 3，范围 0~20；0=禁用工具轮）——深度使用可调大，
# 也可在 TUI 设置屏「工具轮上限」运行时改，或 CLI `agent --tool-rounds N`
MEMAGENT_TOOL_MAX_ROUNDS=3

# 嵌入：SiliconFlow 免费 bge-m3（1024 维）
MEMAGENT_CLOUD_EMBED_URL=https://api.siliconflow.cn/v1
MEMAGENT_CLOUD_EMBED_API_KEY=sk-你的key
MEMAGENT_CLOUD_EMBED_MODEL=BAAI/bge-m3
```

**切换 Anthropic 官方协议（可选）**：聊天/梳理通道支持 `anthropic` 协议——客户端自动改用
`x-api-key` + `anthropic-version` 请求头（非 Bearer）、`system` 顶层独立字段与必填
`max_tokens`，请求打向 `{URL}/messages`：

```ini
MEMAGENT_CLOUD_PROTOCOL=anthropic
MEMAGENT_CLOUD_URL=https://api.anthropic.com/v1
MEMAGENT_CLOUD_API_KEY=sk-ant-你的key
MEMAGENT_CLOUD_MODEL=claude-sonnet-4-5
```

注意：Anthropic 官方**没有嵌入端点**，嵌入通道不随协议切换——须保留上面的 OpenAI 兼容
嵌入段（`MEMAGENT_CLOUD_EMBED_URL` 等指向 SiliconFlow 等服务），否则云端嵌入自动落本地
哈希嵌入。维护通道默认继承 `MEMAGENT_CLOUD_PROTOCOL`，也可用 `MEMAGENT_MAINT_PROTOCOL`
单独指定。

验证配置生效（PowerShell/CMD 直接跑；Git Bash 先 unset 代理变量）：

```
$ python main.py status
LLM: 可用
服务商: chat=cloud(openrouter.ai:stealth/ox-alpha) maint=cloud(api.siliconflow.cn:THUDM/GLM-4-9B-0414) embed=cloud(api.siliconflow.cn:BAAI/bge-m3)
```

```bash
python main.py test          # 逐项诊断五条通道：本地 LM Studio / 主聊天 /models / 主聊天对话 / 维护通道对话 / 云端嵌入（主聊天对话失败时退出码 1）
```

### 2. 写入第一条记忆

```bash
python main.py init
python main.py add "以后给我的回答请保持正式、结构化。" --type preference_statement
```

`--type preference_statement/instruction/identity_statement` 为显式声明，保底入库且可自动取代旧偏好/旧事实。**自我描述（"我是…"开头）会被自动识别为身份声明**，无需手动指定；纯离线（`--no-llm`）时需显式传 `--type identity_statement`。写入时能看到完整管线：

```
门控打分: importance=0.795
情景记忆已写入: #13 [用户要求今后回答需详细完整并附背景解释…]
语义记忆变更: #10 [user] prefers = 回答要详细完整…，取代旧版 [5, 7]
技能已沉淀: 详细完整回答 <- …
```

### 2.5 写入 AI 工作经验（V1.6 绿色通道，零云端等待）

Agent 自己的工作经验与环境状态有专用快车道——门控保底直进长期、同键新值自动取代旧版，
端到端不碰云端 chat（打分/摘要/情感全跳过，技能抽取只试本地 LM Studio、不可用走规则兜底）：

```bash
# 工作经验：--context 任务域必带（同域新经验取代旧版并迭代同名技能的做法）
python main.py add "识图要走 base64 内联，外链会被网关掐断" --type experience --context "识图任务" --outcome success

# 环境状态：最新即正确——同工具名的旧认知被自动取代（版本链保留可回溯）
python main.py add "ffmpeg 已经装好了，直接本地转码" --type env_statement
```

```
门控打分: importance=0.500
情景记忆已写入: #3 [ffmpeg 已经装好了，直接本地转码]
语义记忆变更: #3 [ffmpeg] env_state = ffmpeg 已经装好了…，取代旧版 [2]
```

经验类记忆**低频先忘**（半衰 7 天，约两周不用自动沉底归档），被检索到即续命；
检索时相关技能以 `[技能]` 带出（见下）。分层总表见 `ARCHITECTURE.md` §1.1。

### 3. 检索与回溯

```bash
python main.py retrieve "用户喜欢什么回答风格？"     # 混合检索（FTS 关键词 + 向量余弦）
python main.py history user prefers                  # 偏好版本链：变更史、生效期、被谁取代
python main.py inspect                              # 全量记忆（含摘要记忆溯源、语义状态统计）
python main.py tui                                  # 交互式终端界面（见下）
```

### 3.5 TUI 交互界面（纯标准库，零依赖）

```bash
python main.py tui        # 或 python -m memagent.tui
```

全屏终端界面覆盖日常操作：写入（类型切换 / LLM 开关 / 管线结果面板）、混合检索、
三仓储浏览（`t` 换仓储、`s` 换状态、`Enter` 看详情）、冲突裁决（`a` 采纳新版 / `k` 保留旧版）、
SM-2 复习打分（`0-5`）、睡眠巩固 / 主动遗忘（遗忘需二次确认）、健康报告。

**V1.5 机制在 TUI 里的呈现**（V1.5.1 补齐）：

- **检索页**：命中带标签——working 命中 `[当下]`、联想命中 `[联想]`、
  低置信命中 `[低置信]`（可叠加）、技能带出 `[技能]`（V1.6，豁免低置信——带出跟随触发词）；
  有低置信命中时结果区顶部打 ⚠ 示警行
  （与 CLI 同口径）；每条命中后缀 provenance 溯源 `(证据×N / 验证于X天前[久未验证])`，
  数据与 CLI 的 `build_context` 同源（复用 `ranker.inject_provenance`）。
- **写入页**：管线结果面板如实反映 E3 双写——被门控事件提示「已暂存会话工作记忆
  （本会话内可检索，会话结束蒸发）」/「不入长期记忆（同在本会话工作记忆中）」。
- **三仓储浏览页**：情景记忆详情展示情感唤醒 `arousal`（>0 时标「情感:0.xx」）；
  语义记忆详情展示证据次数 `evidence_count`（真实观测续证，检索不累积）。
- **巩固页**：结果面板展示 NREM 轮数、蒸馏/摘要数、REM 联想明细行、
  今日回忆清单（✓唤回/✗遗忘 + 下次间隔天数）。
- **头部状态行**：新增工作记忆条目数（如 `工作:N`）——TUI 是单进程会话，
  工作记忆在 TUI 里真实存在，随 `r` 刷新。

| 按键 | 作用 |
|---|---|
| `Tab` / `Shift+Tab` / 数字键 | 切换左侧 10 个页面（1 Agent、2 设置、3 总览、4-9 写入/检索/记忆/冲突/复习/整理、0 报告；写入、检索、Agent 三页数字进输入框，启动默认在 Agent 页） |
| `Enter` / `Esc` | 输入页提交 / 清空；列表页展开详情 / 取消确认 |
| 上/下、PgUp/PgDn、Home/End | 列表选择与滚动 |
| `r` | 刷新数据 / 重新探测 LLM（探测在后台线程，不卡界面） |
| `q` / `Ctrl+C` | 退出（输入框内用 `Ctrl+C`） |

排版按全角字符双列计算，中文不错位；慢操作（LLM 写入、巩固）执行前会先绘制"处理中"帧。

检索输出按相关度排序，被取代的旧版本不再出现；每条命中带 provenance 溯源（证据次数 /
距上次验证天数，久未验证的事实会标注），检索不可信时打头示警：

```
- (semantic:0.80) [user] prefers 回答要详细完整,包含背景解释,不为简洁而牺牲信息量 (证据×2 / 验证于3天前)
- (episodic:0.63) 用户要求今后回答需详细完整并附背景解释… (发生于3天前)
- (procedural:0.78) 识图任务｜识图要走 base64 内联，外链会被网关掐断 (技能·成功率100%·用过3次)
⚠ 检索置信度低，以下结果可能不相关（feeling-of-knowing: 不确定）   <- 低置信检索时
- [低置信] (semantic:0.19) … (证据×1 / 验证于120天前)[久未验证]
```

### 3.6 导出 Obsidian 图谱快照（记忆可视化）

```bash
python main.py export-vault "data/vault-snapshot"   # 或任意目录
```

把当前库导出为 Obsidian vault：每个实体一页（active 事实 / 待裁决 / 版本历史 / 邻居）、
每个技能一页（触发/做法/成功率，与任务域实体互链）、每条情景一页（链接到其产出事实的实体）。
三元组值中精确出现的实体名转 `[[链接]]`（与 E4 联想连边同规则），REM 的 `associated_with`
联想自动成边——**用 Obsidian 打开导出目录，自带的图谱视图就是免费的记忆联想可视化**，
零 UI 代码。快照单向：Obsidian 内的修改不会回流记忆库。

### 3.7 Agent 记忆工具调用（V1.7.3）

```bash
python main.py agent                    # 对话中说「帮我查一下记忆里的偏好」即可触发工具
python main.py agent --debug            # 观察：[注入 N 条] / [工具] 调用与结果 / 录入去向
python main.py agent --no-tools         # 关闭工具（只保留自动注入与静默录入）
```

内置对话循环在**自动注入**之外开放了模型点名的记忆工具（`--debug` 可观察全程）：

| 工具 | 作用 |
|---|---|
| `memory_search` | 主动检索记忆——注入漏了时的补查（与 CLI retrieve 同口径，含 E7 不确定示警） |
| `memory_remember` | 点名写入（AI 经验/环境状态；复用 remember()，复述拦截与绿色通道白名单原样生效） |
| `memory_history` | 查某实体的事实版本链 |
| `memory_archive` | 按 id 定向归档一条记忆（用户要求删除时用；只归档不硬删可人工恢复，id 来自检索结果的定位清单） |
| `memory_conflicts` | 只读列出待裁决冲突（**裁决权在用户**，resolve 不对模型开放） |
| `memory_status` / `memory_report` | 库况统计 / 健康报告 |
| `memory_sleep` / `memory_forget` | 睡眠巩固 / 全局遗忘清扫（重操作，仅用户要求时调用；定向删除用 memory_archive） |

实现要点（先注重实用和稳定）：

- **文本协议而非原生 function calling**：模型在回答里输出 `<tool_call>` JSON 块即发起调用，
  适配层 chat 零改动，本地 LM Studio / OpenRouter 免费模型 / Anthropic 协议行为一致；
  解析器容错围栏与裸 JSON、单引号兜底，坏 JSON 回填错误信息让模型自行纠正，不炸循环。
- **工具是补丁不是替代**：每轮框架照旧自动注入记忆，工具轮上限 3 次、结果预算 1200 字；
  工具检索带回的原文并入复述识别参照系——工具通道不是自增强回路的旁路。
- **每次调用落审计**（meta 表 `tool_call`/`tool_result`），会话统计显示工具调用次数；
  未知参数一律拒绝并明说「调用未执行」（杜绝静默丢参 → 模型谎报成功）。
- **首轮不再干等**：CLI 启动即后台预热三通道探测，本地 LM Studio 探测走毫秒级
  socket 预检——首字延迟从 ~10s 降到 ~1s（实测 0.74s）。

### 4. 离线整理（走低配维护通道，不花主模型额度）

```bash
python main.py consolidate   # 多轮睡眠巩固：NREM×3（聚类摘要 + 语义蒸馏 + 冲突处理）+ REM 跨簇联想，自动生成健康报告
python main.py forget        # 主动遗忘：强度重算 → 归档 → 摘要降级 → 硬删
python main.py report        # 随时输出记忆健康报告（状态分布/强度直方图/版本链/REM 联想/审计）
```

```
$ python main.py consolidate
睡眠巩固完成（NREM×3 + REM）: 聚类=3, 蒸馏事实=5, 摘要替代=0, REM联想=2, REM写入=1
  💭 [user] × [邻居王阿姨]（事实 #12 ↔ #15，激活 0.93）
```

人脑一夜 4~6 个睡眠周期，巩固不再是单遍过：每轮 NREM 以上轮蒸馏产出为参照（同键
事实不重复蒸馏），高唤醒（情绪强烈）的记忆优先入簇并作为簇代表保留原文；REM 阶段
在实体激活图上找「图上近邻、嵌入远邻」的跨簇联想对——离线只统计不写，在线才经
LLM 逐对裁决生成低置信 `associated_with` 事实（默认挂起待裁，绝不无证据写记忆）。

### 5. 评测与测试（离线确定性，不花钱）

```bash
python main.py eval          # 22 冲突用例 + 13 场景（含拟人度五场景 + 经验通道场景），核心四指标 + 扩展指标
python main.py eval-mini     # 16 场景迷你版
python -m unittest discover tests   # 201 个测试（默认强制 LLM 离线，零网络依赖）
```

拟人度扩展指标（V1.5 评测先行，**五项全绿**）：保留曲线（SM-2 闭环）、
情感区分度（E2 闪光灯记忆）、会话命中率（E3 工作记忆）、联想召回率（E4 联想激活 +
E5 两跳链式场景扩容 2/2）、元认知校准（E7）。
情感加权：写入时规则 + LLM 双模打分 valence/arousal，高唤醒既加强度分项又延长新近度半衰期
（30 天 ×(1+0.5·arousal)，最多 45 天）——情绪事件先天更强且抗遗忘。巩固输出含
「今日回忆清单」：SM-2 到期事实在睡眠巩固期自提示重演——唤回则间隔增长，唤不回则重置等待下一轮。
工作记忆：门控之前双写进纯内存 scratchpad（容量 9，salience = 重要度 + 唤醒 + 新近三要素，
读取时现算衰减），检索时置顶压过历史记忆——被门控拦下的瞬时上下文（如「这轮对话在整理什么」）
会话内可召回，进程退出自然蒸发，真·瞬时记忆。
联想激活：检索时把语义事实三元组现建成实体图（实体为节点、值中精确出现的实体名为边），
从直接命中的实体出发两跳扩散（激活 = 基础强度 + Σ 源激活 × 0.5）——被激活实体下
未直接命中的事实作「联想命中」追加返回（问「家里养的猫要注意什么」，经 user 实体带出「猫毛过敏」；
问「家里宠物的照料经验」，经 user→邻居王阿姨 两跳带出「她家的狗」），
带 `associated` 标记、只读不写回；直接命中所挂实体被激活的另获小幅排序 boost。
经验通道（V1.6 E8，**4/4 绿**）：`--type experience/env_statement` 绿色类型保底直进长期
（AI 工作经验不是「用户这个人的长期事实」，普通门控天然给低分——点名写入的走快车道，
且零云端等待）；事实键由规则确定性构造（工具名 / 任务域），同键新值自动取代旧版
（环境状态「最新即正确」，版本链保留）；经验层 LFU 分层遗忘（半衰 7 天 + 独立归档阈值
0.35——低频约两周沉底，访问续命）；程序记忆激活——检索按 trigger/policy 文本匹配带出
技能（`[技能]` 标签 + 成功率/使用次数溯源，带出即续命，同域新经验迭代技能做法）。
干扰论遗忘（E6）：检索侧**提取诱发遗忘**——回忆过的事实会小幅抑制同实体、本次未被任何
通道够到的竞争事实（-0.01/次，下限 0.10 不降穿，可被新观测续证反转，被联想带出者不受波及）；
写入侧**线索过载**——同一实体挂靠 active 事实超过 8 条时最旧者每超一条降 0.05
（线索区分度下降）。检索不再累积 evidence_count（evidence 只由真实观测累积，
联想激活 base 的证据项 log 饱和——拆掉检索放大证据的自增强回路）。
元认知 v1（E7，读侧增强）：检索置信度表面化——top 直接命中最高分低于置信阈值时直接与
联想命中标「不确定」（阈值按嵌入后端分档：哈希兜底 0.30 / 真实嵌入 bge-m3 等 0.70，
在线跨主题余弦噪声底实测 0.56~0.64，单档 0.30 会让「不确定」在线永不触发；
working 豁免，CLI 打头示警，只标 meta 不改排序）；provenance
溯源——每条命中带证据次数与距上次验证天数（超 90 天标「久未验证」）；检索完全落空
记入审计并在健康报告出「检索空缺（我知道我不知道）」区块，提示高价值待巩固方向。

## CLI 命令总表

| 命令 | 作用 |
|---|---|
| `init` | 初始化数据库 |
| `tui` | 交互式终端界面（全屏，覆盖写入/检索/浏览/裁决/复习/整理/报告） |
| `agent [--debug] [--no-tools] [--no-stream] [--no-llm] [--context 任务域]` | 记忆原生对话循环：自动注入 + 静默优先录入 + 记忆工具调用（V1.7.3，见 §3.7）；`--debug` 打印注入/录入/工具全程 |
| `add <内容> [--type ...] [--source ...] [--context ...] [--outcome success\|failure] [--no-llm]` | 写入事件（自动门控+编码+冲突消解）。`--type preference_statement/instruction/identity_statement` 为显式声明，保底入库且可自动取代旧偏好；`--type experience/env_statement` 为 AI 经验绿色通道（V1.6，配 `--context 任务域`/`--outcome`，同键快取代 + 零云端等待）；"我是…"式自我描述自动按身份声明处理 |
| `retrieve <问题> [--top-k N]` | 混合检索（FTS 关键词 + 向量余弦 + working 置顶 + 联想命中追加 + 技能带出）；低置信示警 + 每条命中 provenance 溯源（E7） |
| `inspect` | 全量记忆（含摘要记忆的溯源 id、语义记忆状态统计） |
| `history <entity> [relation]` | 事实版本链：value 变更史、生效期、被谁取代、证据次数 |
| `conflicts` / `conflicts resolve <id> --accept-new\|--keep-old\|--both` | 待裁决冲突队列与人工裁决（`--both` = 判定误报：互补事实双方共存，不取代） |
| `consolidate` | 多轮睡眠巩固（NREM×3：聚类摘要 + 蒸馏 + 冲突处理；REM：跨簇联想重组）+ SM-2 到期重演「今日回忆清单」，完成后自动生成健康报告 |
| `forget` | 主动遗忘：强度重算 → 归档 → 摘要降级（原始记忆标 summarized 保留）→ 硬删 |
| `repetition [--review <id> --quality 0-5]` | SM-2 间隔重复队列 |
| `status` | LLM 可用性与三通道服务商、记忆统计 |
| `report` | 输出记忆健康报告（状态分布/强度直方图/版本链/待裁冲突/REM 联想/检索空缺（我知道我不知道）/审计动作） |
| `eval` / `eval-mini` | 评测：偏好命中率 / 冲突消解正确率 / 版本链完整率 / 重复提问稳定率 + 拟人度扩展五项（保留曲线/情感区分度/联想召回率/会话命中率/元认知校准）+ 经验通道（V1.6） |
| `export-vault <目录>` | 导出 Obsidian vault 快照：实体/技能/情景记忆 → markdown 笔记 + `[[wiki 链接]]`，Obsidian 打开即得记忆图谱可视化（只读单向） |
| `reembed [--apply] [--batch-size N]` | 嵌入回填（换嵌入模型后统一维度） |
| `rebuild-fts [--apply]` | FTS 索引重建（中文 2-gram） |
| `alias add/remove/list` | 实体别名表管理 |
| `resolve-entities [--apply]` | 实体归一迁移（幂等） |

> 所有 `--apply` 迁移命令默认 dry-run，执行时自动备份到 `data/backups/` 并记录 `migration_log`。

## LLM 三通道

聊天、梳理、嵌入三通道**各自独立探测与降级，互不影响**。每通道优先级：**本地 LM Studio → 云端 → 离线兜底**（规则打分 / 哈希嵌入）；云端输出格式损坏时按校验器自动尝试 `CLOUD_FALLBACK_MODELS`；云端连续 2 次网络失败自动熔断（置空通道直到探测 TTL 过期——网络黑洞期不再逐次撞网干等，V1.6 起 chat/maint 与 embed 同等待遇）。另有 **`local_chat` 本地快车道**（V1.6）：只试本地 LM Studio、绝不碰云端，供经验绿色通道的技能抽取使用。不配置 `.env` 即纯离线运行；维护通道与嵌入不配置时默认跟随聊天配置。

| 通道 | 环境变量前缀 | 服务对象 | 兜底行为 |
|---|---|---|---|
| 主 `chat` | `MEMAGENT_CLOUD_*` | 门控打分、编码抽取、冲突消解 | 规则打分 |
| 梳理 `maint` | `MEMAGENT_MAINT_*` | 巩固：聚类摘要、语义蒸馏、REM 联想裁决 | 跳过蒸馏 / 首条摘要 / REM 只统计不写 |
| 嵌入 `embed` | `MEMAGENT_CLOUD_EMBED_*` | 全部向量检索与聚类 | 哈希嵌入 |

**免费渠道现状**（免费政策随时可能变化）：

- OpenRouter `stealth/ox-alpha`：输入/输出 $0、1M 上下文，免费档约 50 请求/天（账户曾充值满 $10 约 1000/天）。匿名提供商预发布模型，提示词会被留存（不用于训练），勿发敏感内容
- SiliconFlow `THUDM/GLM-4-9B-0414` / `BAAI/bge-m3`：免费、额度宽松、国内直连

## 依赖与目录

- 纯标准库：`sqlite3`（FTS5）、`urllib`、`dataclasses`、`unittest`；`pyproject.toml` 可安装但无需安装即可运行。
- 向量检索为自写余弦相似度（`core/vectors`），无第三方向量库。

```
005-memory-agent/
├── main.py               CLI 入口（全部命令）
├── ARCHITECTURE.md       架构规范：分层/职责/依赖规则/未来插入点（测试守护）
├── memagent/
│   ├── settings.py       配置（阈值/权重/路径/LLM 三通道，.env 与环境变量可覆盖）
│   ├── pipeline.py       写入主管线（CLI 与评测共用）
│   ├── tui.py            全屏 TUI（msvcrt/termios + ANSI，零依赖）
│   ├── reports.py        记忆健康报告
│   ├── maintenance.py    reembed / 实体归一 / FTS 重建等迁移
│   ├── core/             领域对象 + 时钟/向量/中文分词纯函数
│   ├── adapters/llm/     LLM 抽象 + 本地/云端客户端（三通道唯一入口）
│   ├── attention/        门控（规则 + LLM 混合打分，显式声明保底）
│   ├── encoding/         三类编码 + 实体解析（精确代词映射）
│   ├── memory/           三仓储 + 别名表 + 冲突表 + 审计
│   ├── storage/          SQLite 组合根（schema + 备份 + 迁移日志）
│   ├── retrieval/        混合检索、融合排序、联想激活（实体图扩散）与读侧元认知（低置信表面化/provenance）
│   ├── consolidation/    巩固（并查集聚类 / NREM×3 蒸馏 / REM 联想 / 冲突分级消解）
│   ├── forgetting/       遗忘（强度重算 + 摘要降级）
│   ├── learning/         SM-2 间隔重复 + 强度模型
│   └── eval/             评测（mini golden + 场景 harness）
├── evals/                golden 冲突用例 + 场景文件
├── tests/                201 个单元/集成测试（包级默认 LLM 离线，含迁移回归快照、TUI 排版与架构守护）
├── data/                 数据库 / 备份 / 健康报告（运行时生成）
└── Ultimate-GOAL/        终态期望(D.txt) + 演进方案(PLAN.md) + 项目记录
```

## 设计原则

1. **机制必须可度量**：每个机制配评测场景，证明有用才保留（评测已两次抓出真实缺陷）。
2. **错不如旧**：低置信冲突只挂起（pending）不覆盖，检索继续用旧版。
3. **删除必须保守**：摘要降级保留原始记忆（summarized 状态）双向可溯；任何失败绝不写半成品。
4. **迁移必须可回滚**：dry-run 先行、自动备份、migration_log 可审计。
5. **优先外部记忆，谨慎触碰模型参数**（参数层学习留待证据充分后）。
6. **依赖只许自上而下**：分层与例外见 `ARCHITECTURE.md`，反向依赖会被守护测试拦下。
