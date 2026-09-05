"""全局配置：所有阈值、衰减率、路径集中在此处（V0.2 起随包分发，支持环境变量与 .env 覆盖）。"""
from __future__ import annotations

import os

# 项目根目录 = memagent 包的上一级
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path: str) -> None:
    """极简 .env 加载（纯标准库）：已存在的环境变量不覆盖。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(os.path.join(BASE_DIR, ".env"))

def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    """读整型环境变量并钳位：格式坏/越界回落安全值（.env 是人手编辑的，宁宽进严用）。"""

    def _clamp(v: int) -> int:
        return max(lo, min(hi, v))

    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return _clamp(int(str(raw).strip()))
    except (TypeError, ValueError):
        return default


DATA_DIR = os.environ.get("MEMAGENT_DATA_DIR", os.path.join(BASE_DIR, "data"))
DB_PATH = os.environ.get("MEMAGENT_DB", os.path.join(DATA_DIR, "memory.db"))
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")

# ---- 注意力门控阈值 ----
WRITE_THRESHOLD = 0.50        # 重要性超过该值才进入长期记忆
WORKING_THRESHOLD = 0.20      # 低于工作记忆阈值则丢弃

# 规则打分权重
W_EXPLICIT = 0.35             # 显式指令（"记住""以后"等）
W_RISK = 0.25                 # 风险/安全/合规
W_PREFERENCE = 0.20           # 偏好语句（"喜欢""希望"等）
W_FEEDBACK = 0.20             # 成功/失败反馈

# ---- 遗忘（衰减） ----
FORGET_THRESHOLD = 0.15       # 强度低于该值 -> 归档
ARCHIVE_THRESHOLD = 0.05      # 归档中低于该值 -> 硬删
CONFLICT_PENALTY = 0.20       # 冲突证据的强度惩罚（旧值，保留兼容）

# ---- 强度模型（B3 正式化）----
W_IMPORTANCE = 0.35           # 重要度权重
W_CONFIDENCE = 0.25           # 置信度权重
W_FREQUENCY = 0.20            # 频率权重（log 饱和）
W_RECENCY = 0.20              # 新近度权重（指数半衰）
FREQ_ACCESS_CAP = 2.0         # log(1+access) 封顶（≈7 次访问后频率项满格）
STRENGTH_HALF_LIFE_DAYS = 30.0  # 新近度半衰期（天）
CONFLICT_PENALTY_FACTOR = 0.6   # 冲突未决记忆的强度折扣
# ---- 情感加权（E2 闪光灯记忆）----
W_EMOTION = 0.10              # 唤醒度权重：情绪事件强度分项（加项后五权重和 1.10，
                              # 满格时超出 1 的部分由 strength._clamp01 截断，可解释）
EMOTION_HALF_LIFE_BOOST_K = 0.5  # 高唤醒半衰期延长：×(1+k·arousal)，arousal=1 时 30→45 天

# ---- 冲突消解（B1）----
CONFLICT_AUTO_SUPERSEDE_CONFIDENCE = 0.8  # 模型抽取达到该置信度才允许自动覆盖旧事实
FACT_DEDUP_SIMILARITY = 0.75     # 同键新事实与已有 active 事实的嵌入余弦达到该值视为同一
                                 # 事实的措辞变体——续证旧事实（renew_variant）而非挂冲突。
                                 # 0.75 按中文近义句实测留余量：低于聚类阈值 0.45 的
                                 # "相似"不算，这里要的是"几乎同义"（近义≠同义，宁可
                                 # 漏合并也不误吞真冲突）
# B1 跨键近重复合并：不同 (entity, relation) 键下、嵌入余弦达到该值的 active 事实视为
# 同一知识点的平行变体——写入侧直接吸收进既有事实（不建新行，见 conflict_resolver），
# 存量由 dedupe-facts 治理。0.85 的标定：bge-m3 跨主题噪声底实测 0.56~0.64（E7），
# 同知识不同措辞实测 ≥0.85，阈值落在两带之间留双侧余量。低于阈值一律不合并——
# 宁漏勿错杀：跨键误并会把两条互不冲突的独立事实吞成一条，破坏冲突消解语义
# （取代链与裁决队列的指向都会失真）；漏并的代价只是多一行，dedupe-facts 随时可补。
DEDUP_ABSORB_SIM = 0.85

# ---- 信念分级自动修正（P2-3 bounded belief revision，决策 D4-a）----
# 置信度步进 ±0.2：L2 用户显式反驳对 top-1 相似 active 事实降权、L3 工具结果对同
# 任务域 active lesson 事实步进（success +0.2 封顶 1.0 / failure -0.2），共用本步长。
# 取值依据（D4-a）：一步要「疼但不致命」——0.2 恰是一次续证（upsert renew 取 max）
# 能填平的量，两条同向证据即可反转修正；又远大于 RIF 的 0.01 微扰，区分「证据驱动
# 的信念修正」与「检索侧的抑制噪声」。下限复用 RIF_CONF_FLOOR（降权是可被新证据
# 反转的修正，不是删除，不许降穿）。
# 断路器（安全阀）：双次反向翻转——L3 同任务域连续 2 次工具失败才把整域 lesson
# 信念收回（status 转 pending 待重证，可经检索转正通道挣回来），单次失败只降权；
# 纯 LLM 推演永不改写 active 事实（L4 现状即断路器：model 来源/纯推演/低置信冲突
# 照旧走 pending，修正只由用户显式反驳与真实工具结果两类已证实信号驱动）。
BELIEF_REVISE_STEP = 0.2

# ---- 巩固（睡眠） ----
CONSOLIDATE_MIN_SIMILARITY = 0.45   # 聚类相似度阈值
CONSOLIDATE_MIN_CLUSTER = 2         # 聚类最小成员数（1 个不聚）
CONSOLIDATE_DAILY_LIMIT = 200       # 每次巩固最多处理的事件数
# B5-b 蒸馏低信息量守门：蒸馏 value 去空白后有效长度低于该值直接丢弃（报告计入
# skipped_vague）。实测引子：真实库出现「[系统] 优化 性能与稳定性」这类蒸馏产物
# ——value 仅 6 字，无实体无细节，却被检索续抬（touch_confidence +0.02 无信息量）
# 到 conf=1.0，融合分 0.57 落进 RETRIEVAL_CONFIDENT_BAR_DENSE 的重叠带干扰排序
# （见该常量注释的已知残差）。写入侧掐源头比检索侧打补丁便宜：低信息量事实本就
# 不该入库——情景原文仍在簇里，信息没有丢，只是不配单独立传。只作用于巩固蒸馏
# 通道（LLM 自由输出最容易产出空话），主抽取与显式声明是用户原话，不适用。
DISTILL_MIN_VALUE_CHARS = 8

# ---- 睡眠周期（E5：巩固多轮化 NREM×3 + REM）----
NREM_ROUNDS = 3                     # NREM 轮数：人脑一夜 4~6 个睡眠周期，每轮以上轮蒸馏产出为参照
CONSOLIDATE_MAX_CLUSTER = 8         # 连通分量簇规模上限：传递闭包可能并大簇，超限退回锚点切分
REM_MAX_ASSOC = 10                  # REM 联想对收集与报告上限（「梦」的fragment 数）
REM_ASSOC_MAX_CONFIDENCE = 0.7      # REM 联想事实置信度上限（低置信，且 source=consolidation 绝不自动取代）

# ---- 图式归纳（P2-1：睡眠期把 ≥2 条跨时间的相似情景提炼为一条 relation="rule" 的高阶规律）----
# 断路器语义：rule 是「从多次经历中学到什么」的合成知识，产生门槛必须高于单条蒸馏——
# 三参数同时卡「证据量（几起）× 证据跨度（多久）× 表达上限（多自信）」，任一不满足
# 只统计候选不写库（错归纳不如不归纳，与「错不如旧」同一保守取向）。
RULE_MIN_EPISODES = 2           # 至少几条情景背书才允许归纳（默认 2）：单次事件不成规律
RULE_MIN_SPAN_DAYS = 3          # 组内最早/最晚情景的 created_at 跨度下限（天）：同期巧合
                                # （同一天的多条相似记录）不算跨事件归纳，一律不产 rule
RULE_MAX_CONFIDENCE = 0.7       # rule 置信度上限（LLM 输出钳到 [0, 本值]）：与
                                # REM_ASSOC_MAX_CONFIDENCE 同值——归纳是猜测不是观测，
                                # 一律 pending 走试用期转正通道攒使用证据，绝不自动取代旧 rule
RULE_MAX_PER_NIGHT = 3          # 单夜写入 rule 数上限（B5-c 防抖）：达到后剩余合格组
                                # 只统计不写（rule_candidates 照计、不再请求 LLM）——
                                # 防连续多簇连环产规律刷库；一夜 3 条已是正常人一觉
                                # 能「悟出」的量级，其余的等下一夜继续攒证据

# ---- 摘要降级（B4）----
SUMMARY_DEGRADE_MIN_CLUSTER = 2     # 簇内至少几条才摘要降级
MAX_SUMMARY_SPAN_DAYS = 365         # 簇时间跨度过大不摘要（信息混杂）

# ---- 工作记忆（E3，纯内存会话级）----
WORKING_CAPACITY = 9            # 容量上限（Miller 7±2 取上沿），满则逐 salience 最低者
WORKING_HALF_LIFE_MIN = 30.0    # salience 新近度半衰期（分钟）——比长期记忆的 30 天快三个数量级，这就是「tick」
WORKING_MAX_AGE_HOURS = 24.0    # 超龄即蒸发：会话级信息活不过一天，time_travel 多天场景天然不受干扰
WORKING_W_IMPORTANCE = 0.4      # salience 三权重：重要度
WORKING_W_AROUSAL = 0.3         #   情感唤醒（消费 E2 的 arousal 产出）
WORKING_W_RECENCY = 0.3         #   新近度 exp(-age/半衰)，读取时现算（不开后台线程）
WORKING_RETRIEVE_LIMIT = 3      # 检索时置顶的 working 条数上限（防挤占直接命中的 top_k 名额）

# ---- 检索 ----
RETRIEVE_TOP_K = 5            # 默认返回条数
FTS_WEIGHT = 0.4              # 关键词检索权重
VECTOR_WEIGHT = 0.6           # 向量检索权重
# 分池保底名额（P1-1 第一段·结构修复）：top-k 组装时 episodic 池在「非 working
# 名额」中的保底下限（episodic 池有候选且预算允许时生效；名额只定成员资格，
# 最终仍按分数降序，见 retriever 第 4 步）。
# 为什么需要：semantic 融合分含 confidence 项（旧口径 0.4×conf），巩固蒸馏产物
# conf 普遍 0.8~1.0——跨主题噪声语义事实仅置信项就白拿 0.32~0.4 分、融合分
# 0.67~0.74，永久压过 vec 0.61 的相关情景记忆（episodic 融合实测仅 ~0.52），
# 全局截断后情景记忆系统性失声（真实事故：查询「雨云服务器备份怎么样」）。
# 取 2 的标定：top_k=5 时给情景记忆 2/4 个非 working 席位——1 个挡不住「多条
# 高 conf 噪声占满前列」的挤占（挤占是常态而非个例），3 个则会在语义记忆确实
# 更相关时过度牺牲全局序。与 WORKING_RETRIEVE_LIMIT=3 同一哲学：working 置顶
# 保「当下」的声量，本名额保「往事」的声量，剩余席位仍归全局分数裁决。
EPISODIC_MIN_SLOTS = 2
# semantic 融合权重三项式（P1-1 第二段·权重微调，须第一段落地且十六项评测指标
# 全 100% 后才允许执行；任一指标下降即整体回退本段——既定决策 D6）：
#   fused = VECTOR_WEIGHT×vec + SEMANTIC_CONF_WEIGHT×confidence
#           + SEMANTIC_EVIDENCE_WEIGHT×evidence_component
# 旧口径 0.4×confidence 白送高置信噪声 0.32~0.4 分——巩固蒸馏产物的 conf 普遍
# 0.8~1.0，跨主题噪声语义事实融合 0.67~0.74，实测压过 vec 0.61 的真记忆（融合
# 仅 ~0.52，见 EPISODIC_MIN_SLOTS 注释）。confidence 是「写入时对抽取的自信」，
# 不是「与查询的相关度」——降权到 0.25 后差距由向量项主导；被砍掉的份额由真实
# 证据计数补位：evidence_component = min(1, log1p(evidence_count)/log1p(5))，
# log 饱和与 strength 的频率项（FREQ_ACCESS_CAP）、联想激活 base 的证据加成
# （_EVIDENCE_BASE_BONUS）同一取舍——防「证据多→分高→更易被检索→假证据」的
# 自增强回路，5 次真实观测即满格，之后不再加价。
SEMANTIC_CONF_WEIGHT = 0.25       # 旧值 0.4：高置信噪声的通行证（见上）
SEMANTIC_EVIDENCE_WEIGHT = 0.05   # 证据项：只认真实观测累积的 evidence_count（E6 契约）
# 可检索状态集合：同时驱动两件事——retriever 的结果过滤，与 FTS 索引的写入/删除/重建。
# 两者本就是一件事的两面（索引里放着检索时注定被过滤掉的行 = 白占 search_fts 的 limit 名额）。
# 单点定义的理由（V1.7 P0-1）：此前写入路径无条件写 FTS、状态变更不动 FTS、rebuild_fts
# 只重建 active——三处各自正确，合起来不自洽：已失效记忆（archived / summarized /
# deleted / superseded）永久占据召回名额，库越老召回越差；且跑一次 rebuild-fts 会因
# 索引内容突变而改变检索行为（违反「迁移不改变行为」原则）。
# V1.7 P1 已接通转正通道：pending 进本集合后索引与过滤两侧同时生效——
# 试用期事实必须可被检索，否则永远拿不到「被频繁使用」的证据，试用期只进不出
# （这是 V1.7 开工时的死锁，详见 PLAN-V1.7.md §1 F1）。
RETRIEVABLE_STATUSES = ("active", "pending")
# FTS 候选池放大系数：search_fts 的召回量 = 需求量 × 本系数。
# 为什么需要（V1.7 P0-1）：失效记忆（archived / summarized / deleted / superseded）的
# FTS 索引条目**无法安全删除**——本项目的 FTS 表是 external content 表，单条 DELETE
# 与「UPDATE 到空条目」都会损坏索引，且无法自省条目是否存在（实测结论，见仓储
# _sync_fts 文档）。于是失效条目会一直占据 search_fts 的 limit 名额，库越老越多，
# 有效候选越少——这是「老库检索变差」的真实来源。放大候选池让失效条目挤不掉有效
# 候选，retriever 侧再按 status 过滤，用一点冗余换回正确性。
FTS_CANDIDATE_FACTOR = 3

# ---- 深搜（P1-4：用户显式点名的 LLM 加深记忆检索，默认关闭）----
# 深搜 = 先用一次 llm.chat 把查询展开成 DEEP_EXPANSIONS 条假设性线索（改写 +
# 关联概念/上位词 + 相邻场景），每条线索独立跑「FTS+向量」双路召回后按最强
# 证据合并重排——候选收集与融合组装与快搜共用同一套口径（快搜即 cues=[query]
# 的特例），见 retrieval/deep.py。
# 默认关闭：只有 CLI retrieve --deep / 会话 /deep / memory_search 工具 deep
# 参数显式点名才走，自动注入永远不深搜。三条断路器（deep.py 模块注释 + 单测
# 锁定）：展开线索只用于召回绝不入库；结果条目必有库内 id；全程
# boost_access=False（只读探索，不 _reinforce / RIF / touch_usage / SM-2，
# 也不写 retrieval_gap 审计——深搜前后库字节不变）。
DEEP_EXPANSIONS = 2            # 额外线索数：提示词要求 2~3 条，代码截断到该值。
                               # 取 2 与 ACTIVATION_TOP_N / EPISODIC_MIN_SLOTS 同一
                               # 「宁少勿噪」取向——线索越多召回越广，但噪声候选与
                               # embed 调用同比例增长
# 超时不另设：展开的耗时上限 = llm.chat 现有超时（本地 LM_STUDIO_TIMEOUT=30s /
# 云端 CLOUD_TIMEOUT=60s，连续失败熔断照常）——适配层零改动（任务红线）。

# ---- 试用期转正（V1.7 P1：短期记忆 → 频繁使用 → 自动转正）----
# 背景：pending 事实此前被检索过滤挡在门外，永远拿不到「被频繁使用」的证据，
# 转正通道形同虚设。接通后 pending 参与检索，靠下面三个参数控制「进得来、
# 排得后、够数才转正、没人理就退场」。
PENDING_SCORE_PENALTY = 0.75  # 试用期事实的分数折扣：可召回但不与正式事实争位。
                              # 语义融合分 = 0.6·向量 + 0.4·confidence，正式事实
                              # conf≈0.9 时约 0.96，试用期事实 conf≈0.6 时约 0.84——
                              # 光靠置信度差距不足以拉开身位（尤其小库里 pending
                              # 常常是唯一命中），乘 0.75 后约 0.63，稳定落在正式
                              # 事实之后。取值边界：跌到 0.5 会让试用期事实在正常库
                              # 里几乎永不进 top（转正通道又断了，回到原死锁）；
                              # 抬到 0.9 则惩罚形同虚设，低置信蒸馏产物抢占召回位。
                              # 下界另有保护：折扣后仍须高于哈希嵌入噪声线
                              # （RETRIEVAL_CONFIDENT_BAR=0.30），否则「能被想起」
                              # 这个转正前提本身不成立。
PROMOTE_MIN_HITS = 3          # 转正门槛：累计被检索命中 ≥ 该次数即转 active。
                              # 只数「被想起的次数」，不看分——分数已经被
                              # PENDING_SCORE_PENALTY 压过一轮，再卡分数会让试用期
                              # 事实永远差一口气。3 次的标定：1 次可能是碰巧
                              # （小库里向量碰撞噪声 ~0.1 的候选也会进 top）；2 次
                              # 常是同一会话的连续追问（同一话题连问在评测与真实
                              # 对话里都很常见，不算独立证据）；3 次意味着跨三次
                              # 独立检索仍被想起。与 E8 经验层 LFU 用同一把尺子：
                              # 保活线是 ≥2 次访问，转正比保活高一档。
                              # 度量方式取舍（D3）：先用「绝对次数」落地，不用
                              # 「命中率 ≤ 上限」——后者想排除的是「每轮都被命中的
                              # 背景噪音」，但那条直觉未经验证，且需要维护一个
                              # 轮次计数器（跨会话还得持久化）。自增强回路的风险
                              # 由 E6 契约兜底：检索只 touch confidence 不续证
                              # evidence，命中计数不进联想激活 base，刷不满。
                              # P2 实测补充（Agent 循环改变了计数的节奏）：循环里
                              # 每轮都自动注入，于是「被想起」从「有人主动查」变成
                              # 「每轮被动带出」——同话题连续 3 轮即转正（实测：pending
                              # 事实 hit_count 1→2→3，第 3 轮转 active，evidence_count
                              # 全程恒定 1，E6 分账契约成立）。转正变快是通道接通的
                              # 应有之义，但「被动带出」算不算「被使用」值得再标定：
                              # 若观察到转正过早，优先调本值（循环场景可用更高门槛），
                              # 不要去动检索侧的写回——那会同时破坏再巩固与 SM-2。
PROBATION_MAX_DAYS = 30       # 试用期上限：入库超过该天数且**一次都没被命中**的
                              # pending 事实归档（遗忘侧出口，见 forgetting.policies）。
                              # 30 天 = STRENGTH_HALF_LIFE_DAYS，与情景记忆新近度
                              # 半衰期对齐：一个半衰期内无人问津，说明它不是「被
                              # 间歇想起」而是「根本没用」。只归档 hit_count==0 的
                              # ——已被想起过的事实说明有人需要它，只差次数，不该
                              # 因为慢热被清掉（命中不重置计时：重置会让门槛形同虚设）。

# ---- 记忆原生 Agent 循环（V1.7 P2，落点 memagent/agent，入口层）----
# 读侧：框架在每轮 turn 开始时自动 retrieve + build_context 注入（类前额叶主动
# 注入）——注入时机与裁剪都是框架的事，模型不需要「想起来要查记忆」。
# 写侧：静默优先（默认只进工作记忆、检测到信号才入长期）+ 模型可经 remember()
# 绿色通道强制录入（只接受 GREEN_TYPES，其余降级走门控）。
AGENT_INJECT_TOP_K = 5            # 每轮注入的命中条数上限，与 RETRIEVE_TOP_K 同值：
                                  # 注入就是把一次标准检索的结果拼进提示词，不另设
                                  # 一套排序口径（换口径 = 评的与用的不是同一个东西）
AGENT_INJECT_MAX_CHARS = 800      # 注入上下文的字符预算（build_context 的裁剪点）。
                                  # 800 字 ≈ 5~8 条命中行。宁可少而准：被裁掉的记忆
                                  # 只是这一轮没进提示词，不会丢（仍在库里，相关时
                                  # 下一轮照样被检索到）；而噪声混入会直接带偏回答
AGENT_WORKING_INJECT_LIMIT = 3    # 工作记忆在注入里占的名额（retriever 侧置顶上限是
                                  # WORKING_RETRIEVE_LIMIT=3，两值对齐后本裁剪成为
                                  # 兜底而非挤压）。B3 前曾压到 2：哈希嵌入对极短文本
                                  # 有 0.2~0.3 的随机碰撞相似度，working.search 无相
                                  # 似度下限，3 个名额会被无关闲聊占满、长期记忆只剩
                                  # 2 个。B3 注入门槛落地后长期记忆已被相关性筛选，
                                  # 过得了门槛的直接命中本就该有名额——名额挤压反而
                                  # 误伤「再说一次」这类承接轮：上一轮刚注入的事实
                                  # 文本只存在于 working 里，2 名额会被哈希噪声排序
                                  # 挤掉，复述识别随之失灵（拦截数回落）。闲聊轮的
                                  # 噪声由 B3 门槛兜住（长期记忆不进场），working 名额
                                  # 放回检索侧同值（真想压名额应改检索侧，不是这里）
AGENT_HISTORY_TURNS = 6           # 拼进提示词的最近对话轮数（对话历史由框架裁剪，
                                  # 与记忆注入是两笔独立的上下文预算）
INJECT_MIN_VEC_HASH = 0.30      # 注入相关性门槛·哈希兜底档（B3）：直接命中须 FTS
                                # 关键词命中**或**向量相似度 ≥ 该值才注入（FTS 是
                                # 显式词汇证据，绕过向量门槛；working 是当下上下文，
                                # 永远相关，也绕过）。标定与 RETRIEVAL_CONFIDENT_BAR
                                # 同源：哈希嵌入（64 维字符 2-gram）同题 ~0.35+、
                                # 噪声底 0.2~0.3（AGENT_WORKING_INJECT_LIMIT 与
                                # ACTIVATION_VEC_DIRECT_BAR 的同一实测），0.30 压在
                                # 噪声底之上、同题之下。落点在 injector 不在 retriever：
                                # 检索的小库噪声候选照常返回（E7 用 uncertain 表达
                                # 「自知不可信」），注入才做「不够相关就不进上下文」——
                                # 挤占提示词窗口是循环场景特有的代价
INJECT_MIN_VEC_DENSE = 0.65     # 真实嵌入档（bge-m3 等稠密向量，local/cloud 后端）：
                                # 跨主题余弦噪声底实测 0.56~0.64，门槛压在噪声底之
                                # 上；措辞不同但同题（向量不达标）的损失由 FTS 关键
                                # 词通道兜底。双档判定依赖 llm.embed_backend()，与
                                # retriever.confident_bar 同一套后端探测口径
# 自增强防护：注入的内容打 meta.injected 标记，recorder 据此识别「模型在复述刚注入
# 的内容」并拒绝沉淀——否则「注入 → 复述 → 复述成为新事件 → 与原文相似 → 下次两者都
# 命中 → 计数都涨」会形成自增强回路（E6 已经踩过一次同类回路：evidence 由检索累积）。
AGENT_RESTATE_CONTAIN = 0.6       # 复述识别·覆盖率档：助手文本被注入文本覆盖的字符
                                  # 3-gram 比例达到该值即判定为复述。
                                  # 标定（两组实测对照）：逐字复述「我的回答都要用
                                  # 分点列举」对注入句覆盖率 1.00；基于同一记忆的真实
                                  # 回答「你喜欢分点列举的回答」0.43。0.6 落在两者之间
                                  # 且两侧留余量。用 3-gram 而非 2-gram 的理由：2-gram
                                  # 下两者是 0.50 vs 0.43，区分度不够，逐字复制与换
                                  # 措辞重述只有在 3-gram 上才拉得开
AGENT_RESTATE_MIN_CHARS = 6       # 复述识别·整句包含档：归一化后一方是另一方的子串
                                  # 即判复述，但双方都须达到该长度。6 字下限挡掉「对」
                                  # 「好的」这类极短应答——它们几乎必然是某个长句的
                                  # 子串，不限长会误杀正常对话
AGENT_OFFLINE_REPLY = "（离线模式：没有可用的生成模型，以上为记忆系统自动注入的片段）"
AGENT_SYSTEM_PROMPT = (
    # V1.7.2 用户实测校准：旧文案「片段里没有的就如实说不知道」让模型对常识问题
    # 也答「没有相关记忆」，被用户纠正（「你是有世界知识的」）。诚实约束收窄到
    # 它本来要管的对象：用户的个人事实/偏好/历史，而不是通用知识。
    "你是带长期记忆的助手，同时拥有自己的推理能力与世界知识。\n"
    "[记忆片段] 由记忆系统自动注入，是此前确认过的记忆：直接采信、用于作答，"
    "不要复述原文、不要向用户确认「你是否说过」。\n"
    "片段没覆盖的问题，先用你自己的知识与推理正常回答——只有当问题涉及用户的"
    "个人事实、偏好或历史，而片段里确实没有时，才如实说不知道，不要编造记忆。"
    "用户纠正你时，以纠正后的说法为准。"
)

# ---- Agent 工具调用（V1.7.3，落点 agent/toolkit，入口层）----
# 设计取向（用户定的方向：先注重实用和稳定）：
# 1. 文本协议而非原生 function calling——适配层 chat(prompt, system) 只收发纯文本，
#    免费网关与本地模型对原生 tools 参数的支持参差不齐，文本协议对任何后端行为
#    一致、零网络层改动、离线可测。模型在回答里输出 <tool_call> JSON 块即发起调用。
# 2. 工具只是既有稳定函数的薄封装（retrieve / remember / consolidate / run_forgetting /
#    fetch_history / conflicts.fetch_all / 健康报告），不含任何新记忆机制——
#    写入仍走 pipeline 唯一入口，复述拦截与绿色通道白名单原样生效。
# 3. 自动注入仍是主读路径（框架侧，不靠模型自觉）：memory_search 是注入漏了时的
#    主动补查，提示词里明确「注入已覆盖就不要重复查」。
AGENT_TOOLS_ENABLED = True          # 默认开启；CLI --no-tools / AgentLoop(enable_tools=False) 可关
AGENT_TOOL_MAX_ROUNDS = _env_int("MEMAGENT_TOOL_MAX_ROUNDS", 3, 0, 20)
# 每轮对话最多执行的工具调用轮数（默认 3）。标定：覆盖「查→补查→答」链路
# （记忆工具全是毫秒级本地操作，延迟来自生成轮数本身）；封顶防模型无限循环
# 调用，最后一轮起提示词明示直接作答。可设置（P2-4 深度使用反馈）：.env 写
# MEMAGENT_TOOL_MAX_ROUNDS=8、TUI 设置屏「工具轮上限」行（运行时 A 应用即生效）、
# CLI `agent --tool-rounds N` 三处入口同一语义；0 = 禁用工具轮（等价 --no-tools
# 但保留工具提示词差异），上限 20 防失控。
AGENT_TOOL_RESULT_MAX_CHARS = 1200  # 工具结果回填提示词的字符预算。与注入预算
AGENT_TOOL_RESULT_MAX_CHARS = 1200  # 工具结果回填提示词的字符预算。与注入预算
                                    # AGENT_INJECT_MAX_CHARS=800 同量级取稍宽：
                                    # 工具结果是模型显式索取的（主动查询比被动
                                    # 注入更值得占窗口），但同样只许少而准
AGENT_TOOLS_PROMPT_HEADER = (
    "\n\n[工具]\n"
    "你可以调用记忆工具：需要调用时，先用一两句话说明你要做什么，然后单独输出"
    "如下格式的调用块（JSON 必须合法，一次只调一个）：\n"
    "<tool_call>\n"
    '{"name": "工具名", "args": {"参数": "值"}}\n'
    "</tool_call>\n"
    "系统会执行工具并把结果回填给你，然后你再继续回答；不需要工具时直接回答，"
    "不要输出空的 tool_call 块。\n"
    "工具清单：\n"
)   # 工具清单由 agent/toolkit.build_tools_prompt() 从注册表生成——
    # 说明与注册表不许漂移（漂移 = 提示词承诺的工具执行不了）
    # V1.7.4 措辞校准：旧句「块之外不要写别的文字」被模型理解为「调用前不许说话」，
    # 实测导致模型宣布要删记忆却不出调用块、下轮被追问才补（多轮实测抓出）——
    # 改为「先说明、再单独出块」，与流式 UI 的 [工具] 行渲染也吻合

# ---- 受控命令沙箱（P2-4 Grounded Loop：经验由真实退出码淬炼，安全边界 D5-a）----
# 动机：此前记录的「经验」全是纸上谈兵——「备份失败」这条经验是好是坏、是否配置
# 问题，系统无法验证。run_command 工具给模型一个受控只读沙箱：白名单内检查类命令
# 的退出码与输出经 remember() 沉淀为结构化经验（success/failure 由退出码判定），
# 真实反馈是记忆的磨刀石。
# D5-a 决策（最严档起步）：白名单只含**只读检查类**命令；不开放网络写入、不开放
# 文件删除/写入、不开放包安装。等 Grounded Loop 跑稳了再议受控写操作（D5-b 未决）。
# 扩展方式（责任在用户）：往 SANDBOX_ALLOW_PAIRS 里自行加 (首token, 次token) 对——
# 长度 2 的对匹配命令前两个 token、其余 token 是自由参数（如 ("git", "log") 放行
# `git log --oneline -5`）；长度 1 的对只允许该单词单独成命令（如 ("ls",) 不放行
# `ls -la`）。加对之前自查：该命令的**全部**参数形态是否都只读——白名单是按命令
# 前缀放行的，参数不设防（元字符与换行另有关卡），扩展即担责。
# 为什么 python 裸跑 / python -c 不在白名单：-c 等于任意代码执行（os.system 随手
# 就是 shell），裸 python 是交互解释器同样等价任意代码——只放行 --version 这类
# 无副作用的自报家门。同理解释 git 只放 status/log/diff/show 四个**无条件只读**
# 子命令，commit/push/clean 一律不在表；branch/remote 曾入围但被验收轮移除——
# 前缀级放行挡不住 `git remote add` / `git branch -D` 这类仓库元数据写操作
# （不触用户文件但违背 D5-a「只读检查类」定性），要查分支/远端请用
# `git log`/`git show`/`git diff` 或由用户临时扩表担责。
SANDBOX_ALLOW_PAIRS = [
    ("git", "status"), ("git", "log"), ("git", "diff"), ("git", "show"),
    ("ipconfig",), ("systeminfo",), ("tasklist",), ("where",), ("whoami",),
    ("hostname",),
    ("docker", "ps"), ("docker", "images"),
    ("python", "--version"), ("pip", "list"), ("pip", "show"), ("pip", "--version"),
    ("node", "--version"),
]   # (首token, 次token) 白名单；长度 1 的元组 = 只允许该单词单独成命令。
# ls/dir 已移除（实测翻车：它们是 shell 内置命令而非独立可执行文件，Windows 裸进程
# subprocess 必 WinError 2）——目录/文件读取改由原生工具 list_dir/read_file 承担
# （纯 Python 实现、无进程启动、路径 confinement 到工作区，跨平台）；补入的
# ipconfig/systeminfo/tasklist/where/whoami/hostname 是 System32 真 .exe，裸进程可执行
SANDBOX_TIMEOUT = 10          # 秒，超时强杀
SANDBOX_OUTPUT_MAX_CHARS = 2000   # 回填给模型的 stdout/stderr 合并截断点
SANDBOX_CWD = BASE_DIR        # 工作区根：命令的工作目录锁定在项目根，防相对路径漂移
READ_FILE_MAX_CHARS = 4000    # 原生工具 read_file 的单次读取截断点（read-only，无进程）
LIST_DIR_MAX_ENTRIES = 200    # 原生工具 list_dir 的单次列举上限（防巨目录刷屏）

# ---- 元认知（E7：不确定表面化 + 溯源 + 「我知道我不知道」）----
RETRIEVAL_CONFIDENT_BAR = 0.30  # 检索置信阈值·哈希兜底档：离线哈希嵌入的噪声底标定，
                              # top 直接命中最高分低于该值时，直接命中与联想命中标记
                              # meta.uncertain（工程版 feeling-of-knowing——小库无
                              # 相关度下限，噪声级候选仍会进结果，与其静默返回不如
                              # 显式承认不确定。只标 meta，不改分数不改排序）
RETRIEVAL_CONFIDENT_BAR_DENSE = 0.55  # 真实嵌入档（bge-m3 等稠密向量）。P1-2 重标定
                              # （2026-09-05，evals/calibrate_bar.py，57 行候选×14 探针
                              # =798 样本，P1-1 新融合公式下）：跨主题噪声融合参照
                              # （conf=1.0/evidence 顶格保守上界）P50=0.50/P95=0.554/
                              # P99=0.582/max=0.622，现实噪声（conf≤0.95）普遍 <0.55；
                              # 相关查询实测——情景命中 0.584（雨云条目）、语义命中
                              # ~0.76（vec 0.87）。0.55 = 噪声保守上界 P95 之下、
                              # 相关情景实测之上，纯噪声查询整体落入 uncertain 区，
                              # 相关查询留有余量。旧值 0.70 系旧公式（conf 权重 0.4）
                              # 下标定，新公式下会把「相关记忆在场」的查询也整体标
                              # 不确定（实测：雨云查询全 top <0.70 误触发）。换嵌入
                              # 模型/服务商必须重标（运行 evals/calibrate_bar.py）。
                              # 已知残差（上线实测诚实记录）：conf 被检索续抬到 1.0
                              # 的模糊蒸馏事实（如「[系统] 优化 性能与稳定性」）融合分
                              # 0.57，落在 0.55~0.62 重叠带内可使纯噪声查询不标
                              # uncertain——单阈值 + any-hit 判定的结构性极限，根因是
                              # 检索续抬置信度（touch_confidence +0.02 无信息量）；
                              # 待存量模糊蒸馏事实治理（B5 侧）与按 kind 分档阈值
                              # 评估后重标。（local/cloud 后端取该档，按 embed_backend
                              # 分档，见 retrieval.retriever.confident_bar）
PROVENANCE_STALE_DAYS = 90     # provenance 溯源：距上次验证（semantic 按 valid_from、
                              # episodic 按创建时间）超过该天数的事实标注 meta.stale

# ---- 联想激活（E4，ACT-R 式扩散）----
ACTIVATION_DECAY = 0.5        # 每跳激活衰减系数（源激活 × 衰减 = 邻居获得的流入激活）
ACTIVATION_MAX_HOPS = 2       # 扩散跳数上限（两跳：友邻的友邻）
ACTIVATION_TOP_N = 3          # 联想命中最多外送条数（同实体多事实按激活值竞争）
ACTIVATION_MIN = 0.05         # 激活值低于该值的实体，其挂靠事实不外送为联想命中
ACTIVATION_BOOST = 0.05       # 直接命中所挂实体被激活时的排序加成系数（× min(1, 激活值)）
ACTIVATION_VEC_DIRECT_BAR = 0.2  # 向量通道算「真直接命中」的相似度下限（哈希碰撞噪声底约
                              # 0.1 的两倍）。仅用于联想通道的归属判定——区分真命中与噪声
                              # 泄漏；不给全局检索加下限（V0.5 已论证是过拟合）
ACTIVATION_SEED_DEFAULT = 0.5  # P1-3 别名直启种子的激活值。取值夹在中间：低于真直接命中
                              # （直接命中种子的激活值 = 命中分，真实查询相关度通常更高）
                              # 、高于单跳扩散衰减产物（种子 0.5 × ACTIVATION_DECAY=0.25，
                              # 低于 ACTIVATION_SEED_DEFAULT 的别名种子不至于被淹没）——
                              # 别名命中是「查询词与实体名的字面重叠」，相关性弱于真直接
                              # 命中但强于图上扩散，激活值按证据强度排序。与直接命中种子
                              # 同实体时取 max 不叠加（retriever 种子收集的既有惯例）

# ---- 干扰论遗忘（E6：提取诱发遗忘 + 线索过载）----
RIF_PENALTY = 0.01            # 检索诱发抑制：同实体未命中竞争者每次被检索抑制的置信度
RIF_CONF_FLOOR = 0.10         # 干扰降权的置信度下限（RIF 与线索过载共用，不许降穿——
                              # 抑制是可被新证据反转的降权，不是删除）
CUE_OVERLOAD_N = 8            # 同一实体下 active 事实数超过该值视为线索过载
                              # （线索区分度下降，检索时区分不出该想起哪条）
CUE_OVERLOAD_PENALTY = 0.05   # 线索过载降权：最旧事实每超一条降一次的置信度

# ---- 工作经验绿色通道（E8：经验层——分层记忆策略）----
# 经验类情景记忆的 LFU 寿命：半衰 7 天 + 分层归档阈值 0.35。
# 数学背景：绿色类型走门控保底 importance=0.50，强度地板 = (0.35+0.25)×0.5 = 0.30，
# 普通 FORGET_THRESHOLD=0.15 永远够不着——经验层必须用自己的尺子。按 0.35 阈值：
# 0 次访问 ≈14 天沉底；1 次访问（频率项+访问续命）≈7 周；≥2 次访问频率地板 ≥0.41 保活
# ——低频优先淘汰、高频自动强化，正是「进化：快速取代旧知识」的遗忘侧表达。
EXPERIENCE_HALF_LIFE_DAYS = 7.0   # 经验类新近度半衰期（天）；普通记忆 30 天
EXPERIENCE_FORGET_THRESHOLD = 0.35  # 经验类归档阈值；普通记忆 FORGET_THRESHOLD=0.15
EXPERIENCE_SKILL_TOP_N = 2       # 检索时最多带出的程序记忆（技能）条数
# B5-a env_state 值限长（写入侧）：真实库曾出现巨型 env_state——一条 value 塞了
# IP/面板/SSH 密钥全量信息（用户把整台机器的配置贴进一句话）。情景记忆原文不裁
# （分层设计：情景保留原始事件），只限语义事实的 value——(工具名, env_state) 键
# 的 value 是检索与注入的承载面，单条巨值能独占注入预算（AGENT_INJECT_MAX_CHARS=800）
# 还会被 FTS/向量通道反复召回。300 字足够表达「装了什么/什么版本/能不能用」，
# 超限部分情景原文永远可回溯（截断只发生在绿色确定性构键这一处，experience/
# 显式声明不裁——经验与用户原话没有「最新即正确」的取代语义，裁了就真丢了）。
ENV_STATE_MAX_CHARS = 300

# ---- 间隔重复（SM-2 简化版） ----
SM2_MIN_INTERVAL = 1          # 最短复习间隔（天）
SM2_EASE_MIN = 1.3            # 最低难度系数
SM2_EASE_MAX = 3.0            # 最高难度系数：原始 SM-2 无上界且完美回忆每次 +0.1，
                              # 无上限会加速度放大 interval 爆炸（间隔封顶的伴生钳制）
SM2_EASE_INIT = 2.5           # 初始难度系数
SM2_MAX_INTERVAL = 3650       # 最长复习间隔（天）= 10 年。interval=round(interval×ease)
                              # 是几何增长且无上界：ease≥2.5 时约第 16 次成功回忆即让
                              # now+timedelta(days=interval) 溢出 datetime（年 9999
                              # 上限 ≈365 万天）——物理上限保底（Anki 同样有最大间隔），
                              # 历史越界行由 review() 读侧钳制自愈
SM2_IMPLICIT_THRESHOLD = 0.6  # 检索分达到该值视为一次成功回忆（隐式复习入口）
SM2_RECALL_LIMIT = 20         # 巩固期「今日回忆清单」最多重演的事实数

# ---- LLM 适配 ----
LM_STUDIO_URL = os.environ.get("MEMAGENT_LLM_URL", "http://localhost:1234/v1")  # OpenAI 兼容端点
LM_STUDIO_TIMEOUT = 30        # 秒
EMBED_FALLBACK_DIM = 64       # 本地哈希嵌入维度
LLM_MODEL = os.environ.get("MEMAGENT_LLM_MODEL", "")  # 留空由 LM Studio 默认模型处理

# ---- 云端 LLM（OpenAI 兼容网关，聊天与嵌入可分别配置不同服务商）----
# 未设置 CLOUD_API_KEY 时云端客户端不启用；优先级：本地 LM Studio -> 云端 -> 哈希嵌入兜底
CLOUD_URL = os.environ.get("MEMAGENT_CLOUD_URL", "https://api.siliconflow.cn/v1")
CLOUD_API_KEY = os.environ.get("MEMAGENT_CLOUD_API_KEY", "")
CLOUD_MODEL = os.environ.get("MEMAGENT_CLOUD_MODEL", "tencent/Hunyuan-MT-7B")
# 主模型输出校验失败时依次尝试的备用模型（逗号分隔）
CLOUD_FALLBACK_MODELS = os.environ.get("MEMAGENT_CLOUD_FALLBACK_MODELS", "")
# 记忆梳理（巩固/蒸馏/摘要）可单独指向低配模型（默认跟随聊天配置）
CLOUD_MAINT_URL = os.environ.get("MEMAGENT_MAINT_URL", CLOUD_URL)
CLOUD_MAINT_API_KEY = os.environ.get("MEMAGENT_MAINT_API_KEY", CLOUD_API_KEY)
CLOUD_MAINT_MODEL = os.environ.get("MEMAGENT_MAINT_MODEL", CLOUD_MODEL)
# 厂商协议："openai"=OpenAI 兼容（/chat/completions + Bearer，现状零变化）；
# "anthropic"=Anthropic Messages API（/messages + x-api-key + anthropic-version 头，
# system 为顶层独立字段、max_tokens 必填，见 docs.claude.com/en/api/messages）。
# 用 anthropic 时 URL 须指向 https://api.anthropic.com/v1；Anthropic 官方无嵌入
# 端点，嵌入通道不提供协议配置，须单独配 MEMAGENT_CLOUD_EMBED_URL 指向 OpenAI
# 兼容服务（如上面的嵌入段），否则云端嵌入自动落本地哈希嵌入
CLOUD_PROTOCOL = os.environ.get("MEMAGENT_CLOUD_PROTOCOL", "openai")
# 维护通道协议，默认继承主通道（与 CLOUD_MAINT_URL 默认跟随 CLOUD_URL 同理）
CLOUD_MAINT_PROTOCOL = os.environ.get("MEMAGENT_MAINT_PROTOCOL", CLOUD_PROTOCOL)
# 嵌入服务可指向另一家（默认跟随聊天配置）
CLOUD_EMBED_URL = os.environ.get("MEMAGENT_CLOUD_EMBED_URL", CLOUD_URL)
CLOUD_EMBED_API_KEY = os.environ.get("MEMAGENT_CLOUD_EMBED_API_KEY", CLOUD_API_KEY)
CLOUD_EMBED_MODEL = os.environ.get("MEMAGENT_CLOUD_EMBED_MODEL", "BAAI/bge-m3")  # 聊天模型通常无嵌入端点
CLOUD_TIMEOUT = 60            # 秒
CLOUD_EMBED_TIMEOUT = 20      # 嵌入请求超时（载荷小响应快，比聊天收紧；配合适配层
                              # 连续失败熔断，网络黑洞期快速降级哈希兜底而非干等）
