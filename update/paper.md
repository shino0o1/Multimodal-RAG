知识图谱问答领域近期七篇论文综述
以下综述汇集了近期在知识图谱问答（KGQA）与大型语言模型（LLM）融合领域的七篇代表性论文，按照论文的真实标题整理，对每篇论文的研究目的、方法和亮点进行剖析。所有信息均来自论文原文或公开摘要，不包含虚构内容。

FiDeLiS: Faithful Reasoning in Large Language Models for Knowledge Graph Question Answering
研究目的
LLM在复杂推理任务中常出现幻觉和不可靠回答。FiDeLiS旨在让LLM在知识图谱问答中给出可验证的推理路径，提高答案的事实性和可解释性[1]。
核心方法
FiDeLiS提出了两个训练外组件：
•	逐步束搜索和演绎评分（Deductive Verification Beam Search, DVBS）：模型使用束搜索生成候选推理步骤，并通过演绎评分逐步验证每个推理是否符合知识图谱的事实。当推理步骤足以回答问题时即停止搜索[2]。
•	路径检索模块（Path‑RAG）：为了减少束搜索的搜索空间，该模块根据语义相似度和图结构连接性预筛选候选实体与关系，限制每一步中可能扩展的三元组集合[3]。
这种“先检索再验证”策略使模型的每一步推理均有明确证据支撑，大幅降低幻觉发生的可能。
主要亮点
•	无须对LLM微调：FiDeLiS是完全训练外的框架，只需调用LLM作为推理引擎即可。
•	高效且可解释：通过DVBS与Path‑RAG的结合，框架既保证了推理的可验证性，也避免了大规模遍历知识图谱造成的计算开销[4]。
•	跨多个基准的强性能：论文在不同KGQA数据集上验证了FiDeLiS的有效性，证明该方法可以显著提升答案的正确率并提供清晰的推理路径。

Plan Then Retrieve: Reinforcement Learning‑Guided Complex Reasoning over Knowledge Graphs（Graph‑RFT）
研究目的
现有KGQA模型通常假设知识图谱完整，或仅依赖检索工具获取证据；面对不完整图谱和复杂问题时会出现盲目探索或误检索。Graph‑RFT通过“先规划再检索”来让LLM主动决定何时访问图谱、何时查阅外部信息，从而在不完整知识图谱下进行更合理的推理[5]。
核心方法
Graph‑RFT采用两阶段的强化学习微调框架：
1.	链式思考激活阶段（CoT Fine‑Tuning）：利用专门构建的计划‑检索数据集对LLM进行SFT（supervised fine‑tuning），让模型学会把复杂问题分解成有序的子问题，以及决策什么时候使用KG搜索或网页检索[6]。
2.	计划‑检索强化学习阶段：在第一阶段基础上，模型通过强化学习进一步优化检索策略。引入多重奖励函数，既关注答案正确性，又考虑检索步骤的必要性和效率，从而学会在图谱不完整时主动切换KG检索与网络检索[7]。
为了支持这一框架，Graph‑RFT设计了笛卡尔风格的规划模块，将复杂问题分解为逻辑表达式并指导工具调用；同时通过多重奖励让模型学习何时利用外部知识[8]。
主要亮点
•	覆盖意识的检索调度：通过强化学习，模型能够判断知识图谱的覆盖度，并适时切换到网页检索或返回KG搜索，避免无效扩展[5]。
•	无需大型LLM即可高性能：论文实验表明，Graph‑RFT在多个KGQA基准上显著超越以前的方法，即使使用较小的LLM也能实现强大的复杂问题解决能力[9]。

Generate‑on‑Graph: Treat LLM as Both Agent and KG in Incomplete Knowledge Graph Question Answering
研究目的
绝大多数KGQA方法在“完整”知识图谱上评估，其实现实中的图谱往往缺失许多事实。本论文提出不完整知识图谱问答（IKGQA）任务，并提出Generate‑on‑Graph（GoG）框架，让LLM既充当图谱检索代理，又利用自身内在知识生成缺失三元组，以解决知识不足的问题[10]。
核心方法
GoG是一种训练外的思维‑搜索‑生成三阶段框架：
1.	思维（Thinking）：LLM分析问题并决定下一步是继续在图谱中搜索，还是利用内在知识生成新的事实[11]。
2.	搜索（Searching）：使用KG工具（如SPARQL查询）从当前图谱中检索相关三元组。若检索到的信息不足以回答问题，进入下一阶段[12]。
3.	生成（Generating）：LLM结合上下文生成可能缺失的事实三元组，并利用外部验证来确保生成内容可靠[11]。
这种循环的思维‑搜索‑生成过程不断迭代直至收集到足够证据回答问题。[11]
主要亮点
•	同时利用外部和内部知识：通过让LLM生成缺失的事实，GoG充分发挥LLM的世界知识补全作用，克服图谱不完整性。
•	构建IKGQA数据集：论文创建了真实的不完整KGQA测试集，为该研究方向提供了基准[13]。
•	训练外策略：GoG不需要对LLM进行微调，通过规划及生成即可在主流LLM上运行，实验显示其在多个数据集上优于现有方法。

RJE: A Retrieval‑Judgment‑Exploration Framework for Efficient Knowledge Graph Question Answering with LLMs
研究目的
传统的检索型或代理型KGQA方法要么严重依赖检索信息质量，要么依赖昂贵的专有LLM。RJE旨在减少对LLM调用次数并提升效率，使小参数LLM在不牺牲性能的情况下完成复杂KGQA任务[14]。
核心方法
RJE框架分为三阶段：
1.	检索（Retrieval）：使用图谱检索器生成与问题相关的多条推理路径，并通过推理路径排序模块过滤噪声，选取前K条最相关路径[14]。
2.	判断（Judgment）：将选出的推理路径作为提示输入LLM，由LLM判断这些路径是否充分支持答案；若足够则直接生成答案[15]。
3.	探索（Exploration）：若路径不充分，则启动问题分解模块，将复杂问题拆解为子问题并重新检索相关路径；并通过检索辅助探索模块逐步收集更多证据，直到LLM判断信息充足为止[15]。
此外，RJE引入了针对小模型的辅助模块，如推理路径排序、问题分解和检索辅助探索，使较小的LLM也能高效执行上述流程[14]。
主要亮点
•	高效率与低成本：通过在检索之后判断路径充分性，RJE显著减少了LLM调用次数与输入长度[14]。
•	小模型适用：RJE利用模块化设计，让3B和8B规模的开源LLM也能在多个KGQA数据集上达到甚至超过大型专有模型的性能[14]。
•	无需LLM微调：RJE不需要修改LLM参数，可轻松部署在不同LLM之上。

Harnessing Large Language Models for Knowledge Graph Question Answering via Adaptive Multi‑Aspect Retrieval‑Augmentation
研究目的
检索增强的大型语言模型会引入大量噪声，特别是从多层次（实体、关系、子图）检索的背景下，这些冗余信息会干扰LLM的注意力，导致回答不准确。Amar框架提出自适应的多方面检索增强策略，以更精细地筛选和利用来自知识图谱的不同类型知识[16]。
核心方法
Amar包含两个关键模块：
1.	自对齐模块（Self‑alignment）：在检索到实体、关系和子图等多粒度信息后，使用语义自对齐方法发现它们之间的共性，生成统一的嵌入表示，从而减少不同知识片段间的噪声干扰[17]。
2.	相关度门控模块（Relevance Gating）：通过软门控机制计算问题与不同检索信息之间的相关度。模型根据相关度决定哪些信息用于构建提示，哪些应被过滤，从而避免无关内容误导LLM[18]。
此框架在不改变LLM参数的情况下，将多粒度检索结果转换为嵌入并注入提示，提升了LLM的推理能力。
主要亮点
•	多粒度知识融合：利用实体、关系和局部子图的组合，为LLM提供更丰富且相关的背景信息[19]。
•	动态噪声抑制：自对齐和门控模块协同减少噪声，提高提示质量。
•	实验效果突出：在WebQSP和CWQ两个基准上，Amar在准确率上比最优基线提高约1.9%，在逻辑形式生成准确率上提升6.6%，表明该框架能显著改善LLM的知识图谱推理性能[20]。

Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study over Open‑ended Question Answering
研究目的
现有KGQA研究多集中于封闭式问题，而在现实应用中，用户经常提出开放式复杂问题。本文提出OKGQA基准，用于评估融合知识图谱的LLM在开放式问答场景下的可信性，探讨知识图谱是否能减少LLM的幻觉和错误[21]。
核心方法
•	基准构建：OKGQA包含多种开放式问题，涵盖多跳推理、事实推断与解释性回答。作者还构建了OKGQA‑P，模拟知识图谱存在噪声时的情景[22]。
•	评价指标：除了回答正确率，基准引入FActScore和SAFE等衡量幻觉和事实一致性的指标，以评估LLM+KG系统在开放式问答中的可信度[23]。
•	方法对比：论文系统比较了检索式、代理式及融合式（如FiDeLiS、GoG等）在OKGQA上的表现，分析了知识图谱在降低幻觉方面的作用，并探索了在图谱包含错误时性能的变化[24]。
主要亮点
•	关注开放式场景：OKGQA填补了封闭式KGQA基准与真实世界需求之间的空白。
•	全面的可信性评估：通过新指标对回答准确性和幻觉程度进行量化，为未来研究提供了评测标准[23]。
•	数据公开：提供公开数据和代码，方便社区进一步改进LLM+KG的可靠性。

Memory‑augmented Query Reconstruction for LLM‑based Knowledge Graph Reasoning (MemQ)
研究目的
许多LLM‑驱动的KGQA系统会在推理链中混杂工具调用（例如SPARQL查询）和自然语言推理，既影响可读性又容易导致“幻觉式”查询。MemQ提出以记忆增强的查询重构框架，解耦工具调用和知识推理，并减少无意义的工具调用[25]。
核心方法
MemQ通过以下步骤运行：
1.	记忆构建：利用LLM在对话过程中构建一个包含查询意图的自然语言记忆模块，记录已提出的子查询和已知结果[26]。
2.	自然语言推理：LLM根据记忆在自然语言中进行推理、生成下一步查询的描述，并与图谱交互查询；从而保持推理链的可读性并减少错误调用[27]。
3.	查询重构：模型根据记忆总结现有信息并重构完整查询，从而在必要时调用查询工具；此操作与自然语言推理阶段交替进行[27]。
通过将查询意图显式存放在记忆中，MemQ将工具调用任务与推理分离，避免模型在记忆不足时做出错误或重复调用。
主要亮点
•	解耦工具与推理：记忆模块让LLM保持对查询状态的长期记忆，提升输出的可读性并减少幻觉式调用[25]。
•	自然语言与查询交替：MemQ在自然语言推理和查询重构之间循环，既方便读者理解，又有效利用工具。
•	实验提升：在WebQSP和CWQ等常用KGQA数据集上，MemQ实现了新的最佳性能，显示了其框架的实用价值[28]。
总结
这七篇论文代表了当前知识图谱问答研究中融合大型语言模型的不同思路：有的通过逐步验证和推理路径检索提高答案可信度（FiDeLiS），有的通过强化学习决定何时检索和规划（Graph‑RFT），有的让LLM在不完整图谱中同时作为“代理”和“知识源”进行补全（GoG），还有的通过三阶段检索‑判断‑探索减少LLM调用次数并支持小模型（RJE）。此外，Amar提出多粒度检索与自适应噪声过滤策略，OKGQA基准聚焦开放式问答的可信性评估，MemQ则通过记忆增强的查询重构提高推理链可读性。整体来看，这些工作表明结合结构化知识和LLM的策略正在迅速演化，未来研究可在推理解释性、效率和可信度之间继续寻求平衡。
________________________________________
[1] [2] [3] [4] FiDeLiS: Faithful Reasoning in Large Language Models for Knowledge Graph Question Answering | OpenReview
https://openreview.net/forum
[5] [6] [7] [8] [9] Plan Then Retrieve: Reinforcement Learning-Guided Complex Reasoning over Knowledge Graphs
https://arxiv.org/html/2510.20691v3
[10] [11] [12] [13] [2404.14741] Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering
https://arxiv.org/abs/2404.14741
[14] [15] RJE: A Retrieval-Judgment-Exploration Framework for Efficient Knowledge Graph Question Answering with LLMs
https://arxiv.org/pdf/2510.01257
[16] [17] [18] [19] [20] [2412.18537] Harnessing Large Language Models for Knowledge Graph Question Answering via Adaptive Multi-Aspect Retrieval-Augmentation
https://arxiv.org/abs/2412.18537
[21] [22] [23] [24] [2410.08085] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering
https://arxiv.org/abs/2410.08085
[25] [26] [27] [28] [2503.05193] Memory-augmented Query Reconstruction for LLM-based Knowledge Graph Reasoning
https://arxiv.org/abs/2503.05193
