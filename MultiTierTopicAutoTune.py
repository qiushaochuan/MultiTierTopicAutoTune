# -*- coding: utf-8 -*-
"""
多轮 LDA 主题分析 + 综合报告 (Markdown) + NLTK 常见停用词过滤 + 自动调参示例
-----------------------------------------------------------------
主要改动内容：
  1) 第一轮固定使用最佳参数组合：K=12, alpha=0.1, eta=0.1；
  2) 保持其他轮次网格搜索，并输出各组合评估结果到 JSON；
  3) 为适应约5万条大数据量，每次LDA训练设 chunksize=500；
  4) 仅输出 JSON 文件，不输出其他格式；
  5) 增强代码健壮性，增加异常捕获和详细日志输出。
"""

import os
import json
import uuid
import logging
import datetime
import numpy as np
from tqdm import tqdm
from gensim import corpora, models
import nltk

# 如有需要，可取消下面三行注释下载依赖：
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.coherencemodel import CoherenceModel

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


##############################################################################
# 1. 基础功能函数
##############################################################################

def load_data_from_folder(folder_path):
    """
    遍历文件夹内所有 JSON 文件，将帖子数据合并。
    对于每篇帖子，提取：
      - 文本 = thread_title + 所有 replies.content 拼接
      - doc_id = thread_id（若无则自动生成）
      - doc_url = thread.url
    通过拼接后的文本去重。
    返回：texts, doc_ids, doc_urls, original_docs
    """
    all_texts = []
    all_doc_ids = []
    all_doc_urls = []
    all_original_docs = []
    seen_texts = set()

    try:
        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith(".json"):
                continue
            file_path = os.path.join(folder_path, file_name)
            logging.info(f"正在读取文件: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logging.error(f"读取文件 {file_path} 失败: {e}")
                continue

            for thread in tqdm(data, desc=f"Loading {file_name}", leave=False):
                thread_title = thread.get("thread_title", "")
                thread_url = thread.get("url", "")
                replies = thread.get("replies", [])
                replies_content = [r.get("content", "") for r in replies if r.get("content", "")]
                combined_text = thread_title + "\n" + "\n".join(replies_content)
                if combined_text in seen_texts:
                    continue
                seen_texts.add(combined_text)
                doc_id = thread.get("thread_id", str(uuid.uuid4()))
                all_texts.append(combined_text)
                all_doc_ids.append(doc_id)
                all_doc_urls.append(thread_url)
                all_original_docs.append(thread)
    except Exception as ex:
        logging.error(f"数据加载过程中发生错误: {ex}")
    logging.info(f"合并后共收集帖子数: {len(all_texts)}")
    return all_texts, all_doc_ids, all_doc_urls, all_original_docs


def preprocess_texts(texts):
    """
    对英文文本进行预处理：
      1) 使用 nltk.word_tokenize 分词；
      2) 过滤长度 <=1 的词；
      3) 删除常见停用词；
      4) 单篇文档内部去重。
    返回：[[token1, token2, ...], ...]
    """
    processed_texts = []
    stop_words = set(stopwords.words('english'))
    for text in tqdm(texts, desc="Preprocessing texts"):
        tokens_raw = word_tokenize(text)
        seen_tokens = set()
        filtered_tokens = []
        for token in tokens_raw:
            t_lower = token.lower()
            if len(t_lower) <= 1:
                continue
            if t_lower in stop_words:
                continue
            if t_lower not in seen_tokens:
                seen_tokens.add(t_lower)
                filtered_tokens.append(t_lower)
        processed_texts.append(filtered_tokens)
    return processed_texts


def create_dictionary_and_corpus(processed_texts, no_below=3, no_above=0.8):
    """
    根据预处理后的文本构建 gensim Dictionary 与 BOW 语料，
    并过滤低频和高频词。
    """
    dictionary = corpora.Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    return dictionary, corpus


def run_lda(corpus, dictionary, num_topics, alpha, eta,
            passes=10, iterations=50, random_state=42):
    """
    训练 LDA 模型，设置 chunksize=500 以适应大数据量。
    """
    try:
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            alpha=alpha,
            eta=eta,
            passes=passes,
            iterations=iterations,
            random_state=random_state,
            chunksize=1000
        )
    except Exception as e:
        logging.error(f"LDA 训练失败: {e}")
        raise
    return lda_model


def save_text_file(filepath, content):
    """
    保存文本到文件
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        logging.error(f"保存文件 {filepath} 失败: {e}")


def save_json_file(filepath, data):
    """
    将 Python 对象保存为 JSON 文件
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"保存 JSON 文件 {filepath} 失败: {e}")


def filter_documents_by_topic(lda_model, corpus, threshold=0.2):
    """
    对每篇文档计算主题分布，取最大概率主题，
    若该概率 >= threshold，则将该文档分配至对应主题。
    返回：{topic_id: [doc_index, ...]}
    """
    topic_doc_mapping = {}
    for doc_id, bow in enumerate(tqdm(corpus, desc="Filtering docs by topic")):
        topics = lda_model.get_document_topics(bow)
        if not topics:
            continue
        dominant_topic, prob = max(topics, key=lambda x: x[1])
        if prob >= threshold:
            topic_doc_mapping.setdefault(dominant_topic, []).append(doc_id)
    return topic_doc_mapping


def classify_subtopics(lda_model, corpus, threshold=0.1):
    """
    针对子集 corpus，对每篇文档取概率最大的主题，
    返回：{subtopic_id: [doc_index_in_sub_corpus, ...]}
    """
    mapping = {}
    for i, bow in enumerate(corpus):
        topics = lda_model.get_document_topics(bow, minimum_probability=threshold)
        if not topics:
            continue
        dominant_topic, _ = max(topics, key=lambda x: x[1])
        mapping.setdefault(dominant_topic, []).append(i)
    return mapping


def generate_integrated_report(document_tracker, doc_ids, round1_classification,
                               round2_classification, round3_classification,
                               total_docs, output_folder):
    """
    生成综合报告 Markdown 文件，包含各轮主题分布及文档追踪示例。
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_folder, f"integrated_report_{timestamp}.md")
    lines = []
    lines.append("# 多级主题综合报告\n")
    lines.append(f"总文档数：{total_docs}\n")
    lines.append("---\n")
    lines.append("## Round 1 主题分布\n")
    for topic_key, info in round1_classification.items():
        count = info["sample_count"]
        prop_global = count / total_docs if total_docs else 0
        topic_words = info.get("topic_words", "")
        lines.append(f"- **{topic_key}**: 样本数 = {count}, 占全局 {prop_global:.2%}，主题关键词: {topic_words}\n")
    lines.append("\n## Round 2 主题分布\n")
    for topic_key, info in round2_classification.items():
        count = info["sample_count"]
        parent_topic_key = topic_key.rsplit('.', 1)[0] if '.' in topic_key else topic_key
        parent_count = round1_classification.get(parent_topic_key, {}).get("sample_count", 1)
        prop_global = count / total_docs if total_docs else 0
        prop_parent = count / parent_count if parent_count else 0
        topic_words = info.get("topic_words", "")
        lines.append(f"- **Round2_{topic_key}**: 样本数 = {count}, 占全局 {prop_global:.2%}, 占父 {parent_topic_key} {prop_parent:.2%}，主题关键词: {topic_words}\n")
    lines.append("\n## Round 3 主题分布\n")
    for parent_subtopic, subtopics_info in round3_classification.items():
        lines.append(f"### Round2 子主题: {parent_subtopic}\n")
        for subtopic_key, subinfo in subtopics_info.items():
            count = subinfo["sample_count"]
            parent_count = round2_classification.get(parent_subtopic, {}).get("sample_count", 1)
            prop_global = count / total_docs if total_docs else 0
            prop_parent = count / parent_count if parent_count else 0
            topic_words = subinfo.get("topic_words", "")
            lines.append(f"- {subtopic_key}: 样本数 = {count}, 占全局 {prop_global:.2%}, 占父 {parent_subtopic} {prop_parent:.2%}，主题关键词: {topic_words}\n")
    lines.append("\n## Document Tracker 示例\n")
    sample_count = min(5, len(document_tracker))
    example_indices = list(document_tracker.keys())[:sample_count]
    for idx in example_indices:
        r1 = document_tracker[idx]["round1_topic"]
        r2 = document_tracker[idx]["round2_topic"]
        r3 = document_tracker[idx]["round3_topic"]
        lines.append(f"- DocIndex={idx}, DocID={doc_ids[idx]}, Round1={r1}, Round2={r2}, Round3={r3}\n")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
    except Exception as e:
        logging.error(f"写综合报告失败: {e}")
    logging.info(f"Generated integrated report at {report_path}")
    return report_path


def compute_coherence(lda_model, texts, dictionary):
    """
    计算给定 LDA 模型的主题一致性（coherence）。
    """
    logging.info("  [Coherence] 正在计算主题一致性...（可能需要一段时间）")
    try:
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v',
            processes=1
        )
        coherence = coherence_model.get_coherence()
    except Exception as e:
        logging.error(f"计算主题一致性失败: {e}")
        coherence = 0.0
    logging.info(f"  [Coherence] 计算完成, coherence={coherence:.4f}")
    return coherence


def grid_search_lda_params(corpus, dictionary, texts, param_grid,
                           passes=10, iterations=50, random_state=42):
    """
    对 (K, alpha, eta) 进行网格搜索，并用主题一致性（coherence）评估每个组合，
    返回最佳参数、最佳得分以及所有组合的评估结果列表（all_results）。
    """
    best_score = float("-inf")
    best_params = None
    search_records = []
    search_space = []
    for k in param_grid.get('num_topics', [10]):
        for a in param_grid.get('alpha', [0.5]):
            for e in param_grid.get('eta', [0.01]):
                search_space.append((k, a, e))
    total_combos = len(search_space)
    logging.info(f"开始网格搜索, 待测试组合数量: {total_combos}")
    for i, (k, a, e) in enumerate(search_space, 1):
        logging.info(f"[{i}/{total_combos}] 尝试: K={k}, alpha={a}, eta={e} ...")
        try:
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                alpha=a,
                eta=e,
                passes=passes,
                iterations=iterations,
                random_state=random_state,
                chunksize=500
            )
            score = compute_coherence(lda_model, texts, dictionary)
        except Exception as ex:
            logging.error(f"组合 K={k}, alpha={a}, eta={e} 训练失败: {ex}")
            score = 0.0
        search_records.append({
            "K": k,
            "alpha": a,
            "eta": e,
            "coherence": round(score, 4)
        })
        if score > best_score:
            best_score = score
            best_params = (k, a, e)
    if not best_params:
        logging.warning("网格搜索未找到合适参数，使用默认值。")
        best_params = (5, 0.1, 0.01)
        best_score = 0.0
    k_best, a_best, e_best = best_params
    logging.info(f"网格搜索结束, 最优组合: K={k_best}, alpha={a_best}, eta={e_best}, coherence={best_score:.4f}")
    return best_params, best_score, search_records


##############################################################################
# 2. 主流程：合并所有 JSON 数据并进行多轮主题分析
##############################################################################

def main():
    """
    将指定目录下所有 JSON 文件合并为一个数据集，并进行多轮主题分析。
    最终结果保存到 OUTPUT_FOLDER 下。
    """
    INPUT_FOLDER = r"D:\ORG\org论坛全数据库\org论坛全数据库"  # 请根据实际情况修改
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    global_output_folder = os.path.join(INPUT_FOLDER, f"多级主题分析结果_{timestamp}")
    os.makedirs(global_output_folder, exist_ok=True)

    texts, doc_ids, doc_urls, original_docs = load_data_from_folder(INPUT_FOLDER)
    total_docs = len(texts)
    if total_docs == 0:
        logging.info("未读取到任何文本数据，程序结束。")
        return

    processed_texts = preprocess_texts(texts)
    dictionary, corpus = create_dictionary_and_corpus(processed_texts, no_below=3, no_above=0.8)

    document_tracker = {i: {"round1_topic": None, "round2_topic": None, "round3_topic": None} for i in range(total_docs)}

    # ================= Round 1 =================
    logging.info("=== Round 1 LDA ===")
    round1_folder = os.path.join(global_output_folder, "Round_1")
    os.makedirs(round1_folder, exist_ok=True)

    # 第一轮直接使用固定最佳参数：K=12, alpha=0.1, eta=0.1
    best_k_r1, best_alpha_r1, best_eta_r1 = 12, 0.1, 0.1
    logging.info(f"Round1 固定使用参数: K={best_k_r1}, alpha={best_alpha_r1}, eta={best_eta_r1}")

    try:
        lda_r1 = run_lda(corpus, dictionary,
                         num_topics=best_k_r1,
                         alpha=best_alpha_r1,
                         eta=best_eta_r1,
                         passes=20,
                         iterations=100,
                         random_state=42)
    except Exception as e:
        logging.error(f"Round1 LDA 训练失败: {e}")
        return

    topics_info_r1 = dict(lda_r1.print_topics(num_words=20))
    topic_doc_mapping_r1 = filter_documents_by_topic(lda_r1, corpus, threshold=0.2)

    round1_topic_names = {t: str(t + 1) for t in range(best_k_r1)}
    round1_classification = {}
    for topic_id, doc_indices in topic_doc_mapping_r1.items():
        topic_name = round1_topic_names.get(topic_id, f"Topic{topic_id}")
        for idx in doc_indices:
            document_tracker[idx]["round1_topic"] = topic_name
    for t in range(best_k_r1):
        topic_name = round1_topic_names.get(t, f"Topic{t}")
        indices = topic_doc_mapping_r1.get(t, [])
        sample_count = len(indices)
        folder_t = os.path.join(round1_folder, f"Topic_{topic_name}")
        os.makedirs(folder_t, exist_ok=True)
        unique_samples = []
        for i in indices:
            unique_samples.append({
                "doc_id": doc_ids[i],
                "text": texts[i],
                "url": doc_urls[i]
            })
        samples_path_json = os.path.join(folder_t, f"round1_Topic_{topic_name}_samples.json")
        save_json_file(samples_path_json, unique_samples)
        topic_str = topics_info_r1.get(t, "No report").replace("\n", " ")
        round1_classification[topic_name] = {
            "sample_count": sample_count,
            "folder": folder_t,
            "doc_indices": indices,
            "topic_words": topic_str
        }

    # ================= Round 2 =================
    logging.info("=== Round 2 LDA ===")
    round2_folder = os.path.join(global_output_folder, "Round_2")
    os.makedirs(round2_folder, exist_ok=True)

    lda_round2_models = {}
    round2_classification = {}
    param_grid_r2 = {'num_topics': [3, 5, 10], 'alpha': [0.1, 0.5], 'eta': [0.01, 0.1]}
    passes_r2 = 10
    iterations_r2 = 50

    for topic_id, doc_indices in topic_doc_mapping_r1.items():
        if len(doc_indices) < 30:
            logging.info(f"Round2: Topic {topic_id} 样本太少({len(doc_indices)}), 跳过...")
            continue
        parent_topic_name = round1_topic_names.get(topic_id, f"Topic{topic_id}")
        subset_texts = [processed_texts[i] for i in doc_indices]
        sub_dictionary = corpora.Dictionary(subset_texts)
        sub_dictionary.filter_extremes(no_below=3, no_above=0.8)
        sub_corpus = [sub_dictionary.doc2bow(txt) for txt in subset_texts]
        if len(sub_dictionary.token2id) == 0 or all(len(bow) == 0 for bow in sub_corpus):
            logging.info(f"Round2: Topic {parent_topic_name} 过滤后无有效词汇，跳过...")
            continue

        best_params_r2_sub, best_coherence_r2_sub, search_records_r2_sub = grid_search_lda_params(
            corpus=sub_corpus,
            dictionary=sub_dictionary,
            texts=subset_texts,
            param_grid=param_grid_r2,
            passes=passes_r2,
            iterations=iterations_r2,
            random_state=42
        )
        param_sweep_r2_path = os.path.join(round2_folder, f"param_sweep_Round2_{parent_topic_name}.json")
        try:
            if os.path.exists(param_sweep_r2_path):
                with open(param_sweep_r2_path, 'r', encoding='utf-8') as fr:
                    old_data = json.load(fr)
            else:
                old_data = []
        except Exception as e:
            logging.error(f"读取 {param_sweep_r2_path} 失败: {e}")
            old_data = []
        old_data.append({
            "parent_topic": parent_topic_name,
            "search_records": search_records_r2_sub
        })
        save_json_file(param_sweep_r2_path, old_data)
        best_k_r2_sub, best_alpha_r2_sub, best_eta_r2_sub = best_params_r2_sub
        logging.info(f"Round2: Topic {parent_topic_name} 最佳参数: K={best_k_r2_sub}, alpha={best_alpha_r2_sub}, eta={best_eta_r2_sub}, coherence={best_coherence_r2_sub:.4f}")
        try:
            lda_r2 = run_lda(
                sub_corpus, sub_dictionary,
                num_topics=best_k_r2_sub,
                alpha=best_alpha_r2_sub,
                eta=best_eta_r2_sub,
                passes=20,
                iterations=100,
                random_state=42
            )
        except Exception as e:
            logging.error(f"Round2: Topic {parent_topic_name} 训练出错，跳过: {e}")
            continue
        lda_round2_models[parent_topic_name] = {
            "model": lda_r2,
            "dictionary": sub_dictionary,
            "corpus": sub_corpus,
            "doc_indices": doc_indices
        }
        topics_info_r2 = dict(lda_r2.print_topics(num_words=20))
        subtopic_mapping_r2 = filter_documents_by_topic(lda_r2, sub_corpus, threshold=0.2)
        round2_topic_names = {st_id: f"{parent_topic_name}.{st_id + 1}" for st_id in range(best_k_r2_sub)}
        topic_folder = os.path.join(round2_folder, f"Topic_{parent_topic_name}")
        os.makedirs(topic_folder, exist_ok=True)
        for subtid, subidx_list in subtopic_mapping_r2.items():
            subtopic_name = round2_topic_names[subtid]
            for sub_i in subidx_list:
                original_idx = doc_indices[sub_i]
                document_tracker[original_idx]["round2_topic"] = subtopic_name
            sample_count = len(subidx_list)
            st_folder = os.path.join(topic_folder, f"Subtopic_{subtopic_name}")
            os.makedirs(st_folder, exist_ok=True)
            unique_samples = []
            for sub_i in subidx_list:
                original_idx = doc_indices[sub_i]
                unique_samples.append({
                    "doc_id": doc_ids[original_idx],
                    "text": texts[original_idx],
                    "url": doc_urls[original_idx]
                })
            samples_path_json = os.path.join(st_folder, f"round2_{subtopic_name}_samples.json")
            save_json_file(samples_path_json, unique_samples)
            topic_str = topics_info_r2.get(subtid, "No report").replace("\n", " ")
            round2_classification[subtopic_name] = {
                "sample_count": sample_count,
                "folder": st_folder,
                "doc_indices": [doc_indices[x] for x in subidx_list],
                "local_indices": subidx_list,
                "topic_words": topic_str
            }

    # ================= Round 3 =================
    logging.info("=== Round 3 LDA ===")
    round3_folder = os.path.join(global_output_folder, "Round_3")
    os.makedirs(round3_folder, exist_ok=True)

    round3_classification = {}
    param_grid_r3 = {'num_topics': [5, 8, 3], 'alpha': [0.1, 0.5], 'eta': [0.01, 0.1]}
    passes_r3 = 10
    iterations_r3 = 50

    for parent_topic_name, model_data in lda_round2_models.items():
        lda_r2 = model_data["model"]
        doc_indices_r2 = model_data["doc_indices"]
        for subtopic_name, sub_info in round2_classification.items():
            if not subtopic_name.startswith(f"{parent_topic_name}."):
                continue
            local_indices = sub_info.get("local_indices", [])
            if len(local_indices) < 30:
                logging.info(f"Round3: 子主题 {subtopic_name} 样本太少({len(local_indices)}), 跳过...")
                continue
            sub_texts = [processed_texts[doc_indices_r2[i]] for i in local_indices]
            sub_dictionary_r3, sub_corpus_r3 = create_dictionary_and_corpus(sub_texts, no_below=3, no_above=0.8)
            if len(sub_dictionary_r3.token2id) == 0 or all(len(bow) == 0 for bow in sub_corpus_r3):
                logging.info(f"Round3: 子主题 {subtopic_name} 过滤后无有效词汇，跳过...")
                continue
            best_params_r3_sub, best_coherence_r3_sub, search_records_r3_sub = grid_search_lda_params(
                corpus=sub_corpus_r3,
                dictionary=sub_dictionary_r3,
                texts=sub_texts,
                param_grid=param_grid_r3,
                passes=passes_r3,
                iterations=iterations_r3,
                random_state=42
            )
            param_sweep_r3_path = os.path.join(round3_folder, f"param_sweep_Round3_{subtopic_name.replace('.', '_')}.json")
            save_json_file(param_sweep_r3_path, search_records_r3_sub)
            best_k_r3_sub, best_alpha_r3_sub, best_eta_r3_sub = best_params_r3_sub
            logging.info(f"Round3: 子主题 {subtopic_name} 最佳参数: K={best_k_r3_sub}, alpha={best_alpha_r3_sub}, eta={best_eta_r3_sub}, coherence={best_coherence_r3_sub:.4f}")
            try:
                lda_r3 = run_lda(
                    sub_corpus_r3, sub_dictionary_r3,
                    num_topics=best_k_r3_sub,
                    alpha=best_alpha_r3_sub,
                    eta=best_eta_r3_sub,
                    passes=20,
                    iterations=100,
                    random_state=42
                )
            except Exception as e:
                logging.error(f"Round3: 子主题 {subtopic_name} 训练失败，跳过: {e}")
                continue
            topics_info_r3 = dict(lda_r3.print_topics(num_words=15))
            subtopic_mapping_r3 = classify_subtopics(lda_r3, sub_corpus_r3, threshold=0.1)
            parent_folder = os.path.join(round3_folder, f"Topic_{parent_topic_name}", f"Subtopic_{subtopic_name}")
            os.makedirs(parent_folder, exist_ok=True)
            round3_topic_names = {st_id: f"{subtopic_name}.{st_id + 1}" for st_id in range(best_k_r3_sub)}
            subtopic_results = {}
            for st_id, local_doc_list in subtopic_mapping_r3.items():
                new_subtopic_name = round3_topic_names[st_id]
                global_indices = [sub_info["doc_indices"][i] for i in local_doc_list]
                for g_idx in global_indices:
                    document_tracker[g_idx]["round3_topic"] = new_subtopic_name
                unique_samples = []
                for g_idx in global_indices:
                    unique_samples.append({
                        "doc_id": doc_ids[g_idx],
                        "text": texts[g_idx],
                        "url": doc_urls[g_idx]
                    })
                st_folder = os.path.join(parent_folder, f"Subtopic_{new_subtopic_name.replace('.', '_')}")
                os.makedirs(st_folder, exist_ok=True)
                samples_path_json = os.path.join(st_folder, f"round3_Subtopic_{new_subtopic_name}_samples.json")
                save_json_file(samples_path_json, unique_samples)
                topic_str = topics_info_r3.get(st_id, "No report").replace("\n", " ")
                subtopic_results[new_subtopic_name] = {
                    "sample_count": len(global_indices),
                    "folder": st_folder,
                    "doc_indices": global_indices,
                    "topic_words": topic_str
                }
            round3_classification[subtopic_name] = subtopic_results

    integrated_report_path = generate_integrated_report(
        document_tracker=document_tracker,
        doc_ids=doc_ids,
        round1_classification=round1_classification,
        round2_classification=round2_classification,
        round3_classification=round3_classification,
        total_docs=total_docs,
        output_folder=global_output_folder
    )

    original_data_path = os.path.join(global_output_folder, "merged_original_data.json")
    save_json_file(original_data_path, original_docs)
    logging.info("所有多级主题分析完成。")
    logging.info(f"最终结果保存在: {global_output_folder}")


if __name__ == "__main__":
    main()
