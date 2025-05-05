项目简介
本仓库提供一套针对 海量（5 万条以上）论坛 / 社媒文本 的 三级递进式 LDA 主题抽取与自动调参框架。
特点：
多轮 LDA：Round‑1 固定经验最佳参数，Round‑2/3 通过网格搜索 + Coherence 自动寻找最优超参数，分层细化主题。
大规模数据友好：训练时 chunksize=1000，并对停用词、高低频词做过滤，显著降低内存占用。
STACK OVERFLOW
可解释输出：每轮自动保存采样文档、关键词、主题占比，以及最终综合 Markdown 报告，便于快速洞察。
完整日志：Python logging 模块实时输出进度与错误，方便排查问题。
PYTHON DOCUMENTATION
环境依赖
依赖	版本参考	说明
Python	≥ 3.8	测试环境 3.8.10
gensim	≥ 4.3	LDA、Coherence API
RADIM ŘEHŮŘEK
RADIM ŘEHŮŘEK
nltk	≥ 3.8	分词、停用词
GIST
NLTK
tqdm	≥ 4.60	终端进度条
GEEKSFORGEEKS
numpy	≥ 1.20	数值运算
安装：
bash
复制代码
pip install gensim nltk tqdm numpy
若首次使用 NLTK：
python - <<'PY'
import nltk, ssl;
try: _create_unverified_https_context = ssl._create_unverified_context; ssl._create_default_https_context = _create_unverified_https_context
except AttributeError: pass
nltk.download("punkt"); nltk.download("stopwords")
PY
数据准备
输入目录结构
复制代码
data_folder/
├── forum_1.json
├── forum_2.json
└── ...
每个 JSON 文件为一个论坛批次，最外层为帖子的数组。
必需字段：thread_title, replies[].content；可选 thread_id, url。脚本会自动拼接标题和回复，并做内容去重。
快速开始
克隆仓库并将数据放入同级文件夹：
bash
复制代码
git clone https://github.com/ / .git
cd  
在脚本顶部修改 INPUT_FOLDER 指向数据目录。
运行：
bash
复制代码
python 3.8全org论坛分析自动调参.py
Round‑1 固定 K=12, α=0.1, η=0.1。
Round‑2 / Round‑3 网格搜索范围见源码 param_grid_r2/r3。
MACHINELEARNINGPLUS.COM
输出结构
复制代码
多级主题分析结果_YYYYMMDD_HHMMSS/
├── Round_1/
│    └── Topic_1/…Topic_12/
├── Round_2/
│    └── Topic_1/Subtopic_1.1 …
├── Round_3/
│    └── Topic_1/Subtopic_1.1/Subtopic_1.1.1 …
├── integrated_report_*.md
└── merged_original_data.json
samples.json ：每主题下随机采样的原文、ID、URL。
param_sweep_*.json ：各子集网格搜索记录，以 c_v coherence 排名。
RADIM ŘEHŮŘEK
STACK OVERFLOW
integrated_report.md ：汇总三轮主题分布与示例文档，便于高层汇报或继续分析。
参数与高级用法
入口函数	关键参数	说明
create_dictionary_and_corpus	no_below, no_above	词频阈值，默认 3 和 0.8 （占比）
RADIM ŘEHŮŘEK
grid_search_lda_params	param_grid, passes, iterations	自定义搜索空间及训练轮次
filter_documents_by_topic	threshold	主题概率阈值，控制文档归属严格度
如需在更大数据集或显存有限机器上运行，可减少 passes/iterations 或调大 chunksize 以平衡速度与精度。
STACK OVERFLOW
结果解读
主题关键词：gensim.LdaModel.print_topics() 返回每主题前 N 个高权重词，用于人工命名。
RADIM ŘEHŮŘEK
coherence：度量主题词一致性；一般 > 0.5 即为可接受模型。
STACK OVERFLOW
多级占比：README 中展示的“占父主题比例”有助于发现长尾细分议题。