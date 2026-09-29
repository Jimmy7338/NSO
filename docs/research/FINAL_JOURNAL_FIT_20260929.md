# P5 前置：期刊范围、发表路径与排版适配

核验日期：2026-09-29（Asia/Shanghai）。范围：最多两本期刊，仅使用出版方/期刊官方资料；不投稿、不联系编辑、不新增科研。本文件对应 [最终成果目标 P5](/root/NSO/docs/research/FINAL_PAPER_THESIS_TARGETS_20260929.md)。

## 1. 默认目标与判断

**本轮完整格式核验目标：Journal of Intelligent & Robotic Systems（JIRS，Springer Nature），原创研究稿，仅准备格式包。RAS 保留为优先常规非 OA 费用路径的备选，现只具备通用 `elsarticle` 稿。** 选择 JIRS 是因为能核实其期刊专属规范；不代表已决定支付 APC 或授权投稿。Root 在补查 RAS 官方指南及代理仍失败后作出此排版决策；不能以通用模板冒充 RAS 完整合规。

费用优先条件仍需保留：RAS 的范围包括自主系统及其模块的理论、计算与实验研究；当前“类别信息如何改变预算受限观察及实际重建”的应用/机制主线可以在这个范围内说明。其官方主页同时提供订阅和 OA 选项，订阅路径不向作者收取出版费；页面所列可选 OA APC 为 USD 3,140（税前）。该价格是访问时快照，正式投稿前复核，不据此承诺所有可选服务均免费。[RAS 官方主页](https://www.sciencedirect.com/science/journal/09218890)、[Elsevier 官方范围说明](https://shop.elsevier.com/journals/robotics-and-autonomous-systems/0921-8890)。

**范围匹配是本项目的判断，不是期刊对本稿创新性、证据充分性或录用的认可。** 官方强调推进自主系统研究；本文需要通过 P1 固定具体贡献，以 P3 的有限补证收尾。CPU 仿真符合计算研究的表达方式，但“没有明确要求实车”不能推导为“仿真稿一定可送审”。不因期刊选择另加实车、大规模训练或完整排行榜任务，也不提供分区、录用概率或审稿时长承诺。

| 项目 | RAS：常规非 OA 备选 | Journal of Intelligent & Robotic Systems：本轮排版目标 |
|---|---|---|
| 范围 | 自主机器人、系统模块、理论/计算/实验 | 智能系统与机器人理论及应用，含建模、规划和制造相关方向 |
| 本稿定位 | 普通研究文章：受控主动观测应用与机制验证 | 原创研究文章：机器人规划/感知应用；不是综述 |
| 与本稿的匹配判断 | 类别选向、反馈、成本与实测表面能形成紧凑问题链；证据外推须受限 | 应用范围适配；同样不能把条件性结果包装为一般优势 |
| 常规非 OA 路径 | 官方列出 Subscription，不收作者出版费 | **无此路径**；2024 年 9 月起转为全 OA |
| 访问时费用 | 可选 OA USD 3,140，另税 | APC GBP 2,290 / USD 3,190 / EUR 2,590，另税；按录用日期定价 |
| 本轮选择 | 通用模板留存；专属规范未核实，不标为完整包 | 按已核实指南制作完整格式包；收费、协议、减免另由作者决定 |

JIRS 的应用范围见 [Aims and scope](https://link.springer.com/journal/10846/aims-and-scope)；全 OA 转换由 [期刊主页](https://link.springer.com/journal/10846)明确公告，APC 与减免条件见 [Fees and funding](https://link.springer.com/journal/10846/how-to-publish-with-us)。不能继续引用其历史 hybrid 状态，也不能预设学校资助或减免资格。

## 2. RAS：已核实规范与未核项分开

### 2.1 可立即采用的出版方规范

- **LaTeX 与交付包：** Elsevier 官方提供 `elsarticle`、数字引用及作者年份引用模板。源码包应包含 TeX、参考文献、图件和所需样式；Editorial Manager 的源文件应放在同一级目录，不能保留本机绝对路径。期刊专属参考文献规则优先于模板默认值。[官方 LaTeX 指南](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions)。
- **图件：** 出版方接受矢量 PDF/EPS、位图 TIFF/JPEG 等；最终交付优先使用现有矢量图，而非网页预览 PNG。[Artwork formats](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-formats-checklist)。通用建议为最终图中文字约 7 pt、上下标不少于约 6 pt；位图按实际印刷尺寸检查半色调 300 dpi、组合图 500 dpi、线图 1000 dpi。它们是出版方通用建议，不是已查明的 RAS 特殊栏宽。[Artwork sizing](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing)。
- **Highlights：** 按通用说明准备 3–5 条、每条不超过 85 字符（含空格），单独 Word 文件；RAS 是否在首次投稿时必需仍待期刊指南核验。[官方 Highlights 说明](https://www.elsevier.com/researcher/author/tools-and-resources/highlights)。
- **数据与代码：** 出版方说明研究材料可包含数据、软件、代码、模型和协议，并要求按具体期刊 Research Data 小节确定共享方式。可先准备真实的数据/代码可用性说明；不能把通用鼓励共享写成“RAS 强制指定某仓库/全部代码开源”。[Research data](https://www-prod.elsevier.com/researcher/author/tools-and-resources/research-data)、[Data statement](https://www-prod.elsevier.com/researcher/author/tools-and-resources/research-data/data-statement)。
- **作者贡献：** 按实际工作准备 CRediT，作者本人确认姓名、贡献和顺序。通讯作者负责确保贡献描述准确且获全体同意。[CRediT 官方说明](https://www.elsevier.com/researcher/author/policies-and-guidelines/credit-author-statement)。

上述“准备”属于可逆的投稿材料整理，不代表已经向出版方作出声明。

### 2.2 期刊指南未成功读取的项目

已尝试 [现行 RAS Guide for Authors](https://www.sciencedirect.com/journal/robotics-and-autonomous-systems/publish/guide-for-authors)、[Elsevier 旧官方入口](https://www.elsevier.com/journals/robotics-and-autonomous-systems/0921-8890/guide-for-authors)、官方站内搜索/可见索引及 author-information-pack/PDF 查询。现行和旧入口返回 HTTP 403；未找到可验证的官方指南 PDF。官方支持文章也明确要求以具体期刊指南判断是否强制使用模板：[Elsevier 模板支持说明](https://service.elsevier.com/app/answers/detail/a_id/5955/supporthub/publishing/~/is-there-a-template-available-for-my-manuscript-file%3F/)。

因此下列项保持**待核实**，不从别的 Elsevier 期刊套用：

| 待核项目 | 当前处理 |
|---|---|
| 单匿名或双匿名审稿 | 不宣称双盲/单盲；先将作者信息与正文分离保存，暂不上传任何版本 |
| 投稿菜单的准确 article type 名称 | 拟用普通 Research article；登录正式投稿前按真实菜单确认 |
| 正文字数/页数、摘要上限、关键词数 | 首次核验时稿约 5,000 词、摘要 190 词、5 个关键词；该数为当时工作版，不宣称已满足期刊硬限 |
| 首投版式、行距、行号及强制模板 | 使用 `elsarticle` 官方 preprint 12 pt 默认作为内部交付版 |
| Highlights、graphical abstract 是否必交 | Highlights 可先备好；不把现图1自动当作 graphical abstract |
| 引用排序、作者缩写、期刊名格式 | 保留核验后的14条参考条目与完整 DOI；`elsarticle-num` 可作为备选默认，不标为 RAS 已核定规则 |
| 数据政策层级、代码/数据公开强制程度 | 真实列明可获得的材料、版本及未公开部分；不得虚构 DOI 或“按要求全部公开” |
| 期刊特有表格、补充材料、声明字段 | 准备可编辑表格和逐项清单；最终以可读取的该刊指南/投稿界面核对 |

这些待核项阻止将 P5 标为“正式投稿包全部合规”，但不阻止继续排版、核图、准备声明和完成科研收尾。无需为此重新启动实验。

## 3. AI 使用声明与当前图1的具体处理

2026-09-29 读取的 Elsevier 政策区分解释性示意图、研究数据图和 graphical abstract：解释性示意图可在人工负责和明确披露下使用 AI；真实研究图不得凭空生成或修改数据，数据可视化须从底层数据可复现生成；通用生成式图像工具不能制作 graphical abstract。文字结构/内容准备和研究代码中的 AI 协助分别需在声明/方法中说明；AI 不能署名为作者。政策还限制生成可识别的他人产品/品牌形象。[现行 Elsevier AI 政策](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals)。

**已完成的适配：** 初次排版审计时图1使用生成式 composite；2026-09-29 主稿已换为用户提供的真实小车透明图与程序绘制的矢量架构，来源在图注中写明。轨迹、网格和误差图继续来自保存数据；没有用生成式图像修复研究结果。旧生成图仅为历史资产，不是本轮投稿主图。

实际 AI 使用超过单纯拼写检查，不能写成“未使用 AI”或只写“语言润色”。建议最终声明分别说明：辅助组织/修改文稿、辅助开发实验/分析/绘图代码、人工如何核验引用与执行结果。工具可暂记 OpenAI Codex；具体模型/版本及图像工具版本从真实记录补齐，未知时不杜撰。只有作者实际完成相应人工核验后，才能确认承担全文责任的终稿声明。上述是本项目的适配判断，不是已经完成的作者签署。

## 4. JIRS 当前排版目标的已核实格式要求

该刊为单匿名审稿；原创稿须有数据可用性声明。官方推荐 Springer Nature LaTeX 模板，要求可编辑源码和 PDF；摘要 150–250 词，关键词 4–6 个，标题层级最多三级，数字方括号引用并提供 DOI。图表顺序编号；矢量字体嵌入，位图线图/半色调/组合图分别至少 1200/300/600 dpi。需真实资金、利益冲突、作者贡献信息；LLM 内容使用需说明，纯辅助文字校订有例外。参见 [JIRS Submission guidelines](https://link.springer.com/journal/10846/submission-guidelines)。

其模板官方入口是 [Springer Nature LaTeX author support](https://www.springernature.com/gp/authors/campaigns/latex-author-support)。已保存页面链接的 December 2024 `sn-jnl` 模板（下节）；保留原样式和许可。该页要求 Snapp 源码可由 pdflatex 编译，不使用自定义字体；原模板要求单一 `.tex` 文稿、图件另存。JIRS 的第一页还要求领域类别号，当前稿适合 (2) 模型/仿真、(5) 机器人运动规划、(8) AI机器人制造应用，属本项目的分类判断。MSC代码的具体选择待作者核实，不编造。

期刊专属 Statements and Declarations 小节要求放在参考文献之后；正文保持最多三级标题、正文内图表及连续编号。图注使用 Fig.，编号后及句末无句点。补充材料用 Online Resource 连续引用，文件名 `ESM_1.pdf` 等；其中作者/单位/通讯邮箱只能在作者确认后填写。图件优先提供嵌入字体的矢量源，同时保留适合 LaTeX 的 PDF；单纯编译通过不能替代最终尺寸字迹核查。作者、基金、利益冲突、贡献和 AI 说明的空字段使当前包仍不是可直接提交的签署终稿。

## 5. 已保存的官方模板及使用边界

- 官方说明页：[Elsevier LaTeX instructions](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions)。
- 官方页面实际链接的 ZIP：[elsarticle.zip](https://assets.ctfassets.net/o78em1y1w4i4/4MpsJHO0MOJ2xZuwGTAbOZ/7bc64af36477c5d6cfce335a1f872363/elsarticle.zip)。
- 本地原字节：[elsarticle_official.zip](/root/NSO/docs/thesis/journal_template_20260929/elsarticle_official.zip)，1,430,199 bytes；SHA256 `0b093093e84db49f99bcc9a7c3f69ed1fb61b0147c6d296427aff7963e7f50f6`。
- [数字引用示例源](/root/NSO/docs/thesis/journal_template_20260929/upstream/elsarticle/elsarticle-template-num.tex)、[官方文档](/root/NSO/docs/thesis/journal_template_20260929/upstream/elsarticle/doc/elsdoc.pdf)、[原 README/许可说明](/root/NSO/docs/thesis/journal_template_20260929/upstream/elsarticle/README)、[下载与逐文件清单](/root/NSO/docs/thesis/journal_template_20260929/SOURCE_MANIFEST.json)。

ZIP 保留 Elsevier 2007–2024 版权及 LPPL 1.3 或更新版本许可。原包不直接包含 `elsarticle.cls`；README 的正规生成步骤是 `latex elsarticle.ins`，从所附 `.dtx` 提取。模板获取阶段未执行模板内脚本、未安装依赖，既有主稿和 builder 未改；新 builder 使用独立文件。 后续在独立适配目录处理生成文件，原包保持不变。通用 `elsarticle` 不等于期刊专属模板，也不使用 CRC 模板冒充 RAS 格式。

### Springer Nature 官方模板（当前目标）

- 来源：[官方 LaTeX 支持页](https://www.springernature.com/gp/authors/campaigns/latex-author-support)链接的 [December 2024 ZIP](https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12)。
- 原包：[sn_jnl_official.zip](/root/NSO/docs/thesis/journal_template_20260929/springer/sn_jnl_official.zip)，901,814 bytes，SHA256 `812e76dcaa9c28dc1bff1fb6065d51729b67d4ea140552a05088317414a3ecae`；16个原文件未修改。
- [sn-article.tex](/root/NSO/docs/thesis/journal_template_20260929/springer/upstream/sn-article-template/sn-article.tex)、[原 sn-jnl.cls](/root/NSO/docs/thesis/journal_template_20260929/springer/upstream/sn-article-template/sn-jnl.cls)、[用户手册](/root/NSO/docs/thesis/journal_template_20260929/springer/upstream/sn-article-template/user-manual.pdf)、[逐文件清单](/root/NSO/docs/thesis/journal_template_20260929/springer/SOURCE_MANIFEST.json)。
- `sn-jnl.cls`保留 LaTeX3版权及 LPPL 1.3c或更新许可，所有原始版权说明均保留；原包没有单列LICENSE，不能另授本项目MIT许可。
- 新脚本将采用官方 `sn-mathphys-num` 数字样式选项，单一主文 TeX 和独立图件。Root 已安装标准 pdfLaTeX 基础/推荐环境；真实 `sn-jnl` 单页探针经三轮 pdfLaTeX 编译通过。模板另需的 cuted/appendix/threeparttable/wrapfig 四个小包共约1.66MB压缩，保留CTAN原始包、许可、生成记录和SHA。新脚本默认 pdfLaTeX，无网络回退；正式主文仍需另行编译和目视审查，不能把单页探针成功当成整稿成功。

## 6. 当前稿件的有限适配清单

审阅输入为 [focused article](/root/NSO/docs/thesis/ARTICLE_SUBMISSION_EN_20260928.md)，读取时 SHA256 `daccbec8605b51d72bc1f897793ab2418f539452de4f860c8495d9487f66d242`；按空白分词 4,992 词、摘要 190 词、8 组图、5 张表、14 条引用。后续 P3 并稿会更新此快照，不影响本次期刊规范核验。

| 项目 | 已有材料与下一步 | 是否要求新科研 |
|---|---|---|
| 定位与标题 | 明确静态工业设施主动观测、类别作用的条件性；不称共享已获益或 SLAM 位姿精度提高 | 否；按 P1/P3 并稿 |
| 正文和补充材料 | 新8条布局验证已并入；当前主稿8图6表，128/36/40与新增8条各自说明单位；完整证据PDF作为Online Resource | 否 |
| 主图 | 已换真实平台图+矢量流程；8组主图来自保存数据，目视检查稿内排版；独立PDF保留放大细节 | 否 |
| 引用 | 已核验14项；13项DOI，WEF保留官方URL；单一TeX内含参考条目、独立references.tex/json备份与原编号映射 | 否 |
| 表格 | 现有最多6列保持可编辑；CELL/G原评价失败和派生值脚注保留 | 否 |
| 数据与代码说明 | 用实际可访问版本链接、归档清单、恢复指令替换 `/root/NSO/...`；未存在的 DOI 留空 | 否 |
| 打包与PDF | 新适配目录扁平化、打包源码/图/文献，离线构建并目视核查；原通用证据稿保留 | 否 |
| 投稿附带材料 | 准备 Highlights、投稿信草稿、资金/利益冲突/CRediT/AI 声明草稿；不虚构“全体作者已同意” | 否 |
| 最后期刊核查 | JIRS按第4节核验；RAS仍保留第2.2节未核项。实际投稿前需作者字段、声明与费用确认 | 否 |

### 作者和学校信息留空

| 字段 | 待作者确认的值 |
|---|---|
| 作者姓名及顺序 | |
| 单位、院系、城市、国家 | |
| 通讯作者及邮箱 | |
| ORCID | |
| 基金及编号 | |
| 利益冲突 | |
| 每位作者 CRediT 贡献 | |
| 最终数据/代码发布版本与永久标识符 | |
| 学校正式名称与论文模板 | |

本文件完成 P5 的范围/费用核验和官方模板准备；**正式投稿仍需作者信息、声明和费用确认；格式构建状态见第7节及最终构建收据**。未发送任何外部消息、未登录投稿系统、未创建投稿。

## 7. 实施记录（主稿最终构建前）

- 独立脚本：[build_final_journal_20260929.py](/root/NSO/scripts/build_final_journal_20260929.py)，默认仅准备；`--build --render --export` 执行三轮 pdfLaTeX、日志检查、PDF渲染与源码ZIP导出。既有builder与实验源未改。
- 主文单一`main.tex`；数字引用由文中首次引用排序生成且保留14条原始条目映射；图件扁平化，表格值原样并同时保存可编辑表源；CELL/G派生值注释随表保留，投稿版脚注改小写上标字母。所有作者字段空置、声明标为待作者确认。
- [6项转换测试](/root/NSO/tests/test_build_final_journal_20260929.py)覆盖数学/代码中的非引用方括号、未知及未引用文献、摘要与关键词限制、原表值/脚注/标题层次保持、外部路径拒绝和源文件变化拒绝。
- [pdfLaTeX模板探针](/root/NSO/tmp/jirs-template-probe_20260929/pdflatex/probe_receipt.json)已通过；它只检验编译依赖，不包含研究结果。
- 本轮没有向任何期刊提交，也没有承诺费用或替作者签署声明。完整图表可读性、正式作者信息、MSC分类、永久数据版本与声明确认仍在最终包清单中逐项显示。

### 最终并稿后格式检查快照

2026-09-29：主稿已含新8条验证、8组主图、6表、14参考；摘要215词、5关键词。旧第6节4992词/5表数字只记录初次读取时点，不是现稿。JIRS主稿三轮pdfLaTeX已通过，20页，0条溢出/未定义引用/缺字诊断；已目视全部20页contact sheets。多面板图的细字仍需按最终印刷尺寸由作者复核，独立矢量PDF随包供放大查看；不将稿内所有细字均宣称达到8pt。

完整参考DOI核验见 [verification.json](/root/NSO/docs/research/reference_doi_20260929/verification.json)。部分条目通过出版社直接登记的Crossref记录及原项目页交叉核实；不是从第三方引文网站猜填。

## 8. 最终交付（2026-09-29）

**已完成 JIRS 投稿材料内部准备；尚未由作者签署或投稿。** 默认输出 [主文PDF](/root/NSO/docs/thesis/submission_20260929/main.pdf)、[单一TeX](/root/NSO/docs/thesis/submission_20260929/main.tex)、[完整50文件ZIP](/root/NSO/docs/thesis/submission_20260929.zip)、[收据](/root/NSO/docs/thesis/submission_20260929/build_receipt.json)、[作者待填字段](/root/NSO/docs/thesis/submission_20260929/author_fields.json)及[包说明](/root/NSO/docs/thesis/submission_20260929/README.md)。

- 主文20页、8组图、6表、14参考；摘要215词、5关键词。13项提供已核实DOI，WEF保留原出版方URL。
- 实际编译器 pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian)，三轮编译；0条Overfull/undefined/Missing-character诊断。
- 全尺寸Fig1–Fig8矢量PDF独立提供，主文内嵌同源图件；Table1–Table6为可编辑源。CELL/G失败与独立派生值脚注完整保留。
- 五份Online Resource中ESM_4为已经并入新8任务的36页完整证据稿；不是此前旧版证据PDF。完整methods作为ESM_5保留。
- 主文PDF 1,617,659 bytes；最终ZIP 14,853,806 bytes、50文件。34项输入、49项导出文件SHA及ZIP全部50文件逐字节一致性核验通过。最终新表6、架构页、失败脚注、路径图和参考页均已目视。
- [投稿信草稿](/root/NSO/docs/thesis/submission_20260929/cover_letter_draft.md)说明题目、贡献、原确认和全部8个新增任务的条件性结果、期刊范围匹配；原创性、未一稿多投、全体作者同意等均明确留待作者确认，未代签。[投稿检查清单](/root/NSO/docs/thesis/submission_20260929/submission_checklist.md)集中作者字段、MSC、声明、费用和实际系统上传步骤，稿件已有声明无需重复。
- 最后仅执行`--refresh-materials`补齐以上材料并重封包，未重编译PDF或更改科学稿。原有45项产物（不含被更新的当前收据）全部保持字节不变；[原编译收据](/root/NSO/docs/thesis/submission_20260929/pdf_build_receipt.json)和[当时builder](/root/NSO/docs/thesis/submission_20260929/builder_at_pdf_compile.py)独立保存，新收据明确区分原编译与本次材料封包。
- 所有处理仅为出版排版/文献元数据核验；没有运行新的World、规划或评价。

**保留的真实待填事项：** 作者/单位/通讯邮箱、基金/利益冲突/贡献、MSC分类、正式公开版本标识、准确AI使用说明及作者确认、补充材料作者信息、正式投稿时的APC或减免决定。这些属于作者信息和外部投稿流程，不构成重新开启科研实验的条件。模板匹配和PDF可读性不等于录用认可；作者未确认前不得直接提交。RAS通用模板仍留存为费用路径备选，其403导致的专属规范待核状态没有改变。
