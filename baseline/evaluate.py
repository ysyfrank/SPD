import os
import json
import gzip
import numpy as np
from tqdm import tqdm


def average_precision(truth, pred):
    """
        Computes the average precision.
        
        This function computes the average precision at k between two lists of items.

        Parameters
        ----------
        truth: set
                    A set of ground-truth elements (order doesn't matter)
        pred: list
                    A list of predicted elements (order does matter)
        Returns
        -------
        score: double
                    The average precision over the input lists 
    """
    if not truth:
        return 0.0
    
    score, hits_num = 0.0, 0
    for idx, doc in enumerate(pred):
        if doc in truth and doc not in pred[:idx]:
            hits_num += 1.0
            score += hits_num / (idx + 1.0)
    return score / max(1.0, len(truth))


def NDCG(truth, pred, use_graded_scores=False):
    score = 0.0
    for rank, item in enumerate(pred):
        if item in truth:
            if use_graded_scores:
                grade = 1.0 / (truth.index(item) + 1)
            else:
                grade = 1.0
            score += grade / np.log2(rank + 2)
    
    norm = 0.0
    for rank in range(len(truth)):
        if use_graded_scores:
            grade = 1.0 / (rank + 1)
        else:
            grade = 1.0
        norm += grade / np.log2(rank + 2)
    return score / max(0.3, norm)


def metrics(truth, pred, metrics_map):
    """
        Return a numpy array containing metrics specified by metrics_map.
        truth: set
                    A set of ground-truth elements (order doesn't matter)
        pred: list
                    A list of predicted elements (order does matter)
    """
    out = np.zeros((len(metrics_map),), np.float32)

    if "MAP@20" in metrics_map:
        avg_precision = average_precision(truth, pred[:20])
        out[metrics_map.index('MAP@20')] = avg_precision
    
    if "P@1" in metrics_map: # 第1个结果是否命中
        intersec = len(truth & set(pred[:1]))
        out[metrics_map.index('P@1')] = intersec / max(1., float(len(pred[:1])))

    if "P@10" in metrics_map: # 前10个返回的结果中有多少个命中的
        intersec = len(truth & set(pred[:10]))
        out[metrics_map.index('P@10')] = intersec / max(1., float(len(pred[:10])))
    
    if "P@20" in metrics_map: # 前20个返回的结果中有多少个命中的
        intersec = len(truth & set(pred[:20]))
        out[metrics_map.index('P@20')] = intersec / max(1., float(len(pred[:20])))
        # print(intersec, max(1., float(len(pred[:20]))))

    if "P@100" in metrics_map: # 前100个返回的结果中有多少个命中的
        intersec = len(truth & set(pred[:100]))
        out[metrics_map.index('P@100')] = intersec / max(1., float(len(pred[:100])))
    
    if "MRR" in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred):
            if item in truth:
                score = 1.0 / (rank + 1.0)
                break
        out[metrics_map.index('MRR')] = score
    
    if "MRR@100" in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred[:100]):
            if item in truth:
                score = 1.0 / (rank + 1.0)
                break
        out[metrics_map.index('MRR@100')] = score
    
    if "MRR@10" in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred[:10]):
            if item in truth:
                score = 1.0 / (rank + 1.0)
                break
        out[metrics_map.index('MRR@10')] = score
    
    if "NDCG@100" in metrics_map:
         out[metrics_map.index('NDCG@100')] = NDCG(truth, pred[:100])

    if "NDCG@20" in metrics_map:
         out[metrics_map.index('NDCG@20')] = NDCG(truth, pred[:20])

    if "NDCG@10" in metrics_map:
         out[metrics_map.index('NDCG@10')] = NDCG(truth, pred[:10])
    
    return out


class evaluator:
    def __init__(self):
        self.METRICS_MAP = ['MRR@100', 'MRR', 'NDCG@100', 'NDCG@20', 'NDCG@10', 'MAP@20', 'P@1', 'P@10', 'P@20', 'P@100']
    
    def evaluate_ranking(self, docid_truth, all_doc_probs, doc_idxs=None, query_ids=None, match_scores=None, operation="generate_doc"):
        mrr_list = []
        mrr_100_list = []
        ndcg_10_list = []
        ndcg_20_list = []
        ndcg_100_list = []
        map_list = []
        p_1_list = []
        p_10_list = []
        p_20_list = []
        p_100_list = []

        if operation == "top_generate_doc":
            for docid, probability, doc_idx in tqdm(zip(docid_truth, all_doc_probs, doc_idxs)):
                click_doc = set([docid])
                probs = [[doc_idx[idx], probability[idx]] for idx in range(len(probability))]
                probs = sorted(probs, key=lambda x:x[1], reverse=True)

                sorted_docs = [probs[idx][0] for idx in range(len(probs))]
                _mrr100, _mrr, _ndcg100, _ndcg20, _ndcg10, _map20, _p1, _p10, _p20, _p100 = metrics(truth=click_doc, pred=sorted_docs, metrics_map=self.METRICS_MAP)
                # print(f"click_doc: {click_doc}, mrr100: {_mrr100}, ndcg100: {_ndcg100}")
                
                mrr_100_list.append(_mrr100)
                mrr_list.append(_mrr)

                ndcg_100_list.append(_ndcg100)
                ndcg_20_list.append(_ndcg20)
                ndcg_10_list.append(_ndcg10)
                
                p_1_list.append(_p1)
                p_10_list.append(_p10)
                p_20_list.append(_p20)
                p_100_list.append(_p100)

                map_list.append(_map20)

        if operation == "generate_doc":
            for docid, probability in tqdm(zip(docid_truth, all_doc_probs)):
                click_doc = set([docid])
                probs = [[idx, probability[idx]] for idx in range(len(probability))]
                probs = sorted(probs, key=lambda x:x[1], reverse=True)

                sorted_docs = [probs[idx][0] for idx in range(len(probs))]

                _mrr100, _mrr, _ndcg100, _ndcg20, _ndcg10, _map20, _p1, _p10, _p20, _p100 = metrics(truth=click_doc, pred=sorted_docs, metrics_map=self.METRICS_MAP)
                
                mrr_100_list.append(_mrr100)
                mrr_list.append(_mrr)

                ndcg_100_list.append(_ndcg100)
                ndcg_20_list.append(_ndcg20)
                ndcg_10_list.append(_ndcg10)
                
                p_1_list.append(_p1)
                p_10_list.append(_p10)
                p_20_list.append(_p20)
                p_100_list.append(_p100)

                map_list.append(_map20)

        elif operation == "retrieve_doc":
            last_queryid = -1
            click_doc, probs = [], []
            for idx, (label, queryid, score) in zip(docid_truth, query_ids, match_scores):
                if queryid != last_queryid:
                    if click_doc != []:
                        click_doc = set(click_doc)
                        probs = sorted(probs, key=lambda x:x[1], reverse=True)
                        sorted_docs = [probs[idx][0] for idx in range(len(probs))]
                        
                        _mrr100, _mrr, _ndcg100, _ndcg20, _ndcg10, _map20, _p1, _p10, _p20, _p100 = metrics(truth=click_doc, pred=sorted_docs, metrics_map=self.METRICS_MAP)

                        mrr_100_list.append(_mrr100)
                        mrr_list.append(_mrr)

                        ndcg_100_list.append(_ndcg100)
                        ndcg_20_list.append(_ndcg20)
                        ndcg_10_list.append(_ndcg10)
                        
                        p_1_list.append(_p1)
                        p_10_list.append(_p10)
                        p_20_list.append(_p20)
                        p_100_list.append(_p100)
                        map_list.append(_map20)
                    click_doc, probs = [], []
                    last_queryid = queryid
                probs.append((idx, score))
                if label == 1:
                    click_doc.append(idx)
                        
            click_doc = set(click_doc)
            probs = sorted(probs, key=lambda x:x[1], reverse=True)
            sorted_docs = [probs[idx][0] for idx in range(len(probs))]
            
            _mrr100, _mrr, _ndcg100, _ndcg20, _ndcg10, _map20, _p1, _p10, _p20, _p100 = metrics(truth=click_doc, pred=sorted_docs, metrics_map=self.METRICS_MAP)

            mrr_100_list.append(_mrr100)
            mrr_list.append(_mrr)

            ndcg_100_list.append(_ndcg100)
            ndcg_20_list.append(_ndcg20)
            ndcg_10_list.append(_ndcg10)
            
            p_1_list.append(_p1)
            p_10_list.append(_p10)
            p_20_list.append(_p20)
            p_100_list.append(_p100)
            map_list.append(_map20)

        return [np.mean(mrr_100_list), np.mean(mrr_list), np.mean(ndcg_100_list), np.mean(ndcg_20_list), np.mean(ndcg_10_list), np.mean(map_list), np.mean(p_1_list), np.mean(p_10_list), np.mean(p_20_list), np.mean(p_100_list)]

    def evaluate_relevant_psg(self, truth_probs, test_file, evaluate_output):
        fout = open(evaluate_output, "w")
        test_lines = open(test_file).readlines()
        one_doc = []
        last_docid = -1
        for idx in range(len(truth_probs)):
            line = json.loads(test_lines[idx])
            tokens = line["input_tokens"]
            doc_id, doc_score = truth_probs[idx]
            if doc_id != last_docid:
                if one_doc != []:
                    one_doc = sorted(one_doc, key=lambda x:x[1], reverse=True)
                    for did, score, content in one_doc:
                        fout.write(str(did) + "\t" + str(score) + "\t" + content + "\n")
                one_doc = []
                last_docid = doc_id
            one_doc.append([doc_id, doc_score, ' '.join(tokens)])
        
        if one_doc != []:
            one_doc = sorted(one_doc, key=lambda x:x[1], reverse=True)
            for did, score, content in one_doc:
                fout.write(str(did) + "\t" + str(score) + "\t" + content + "\n")
        
        fout.close()

    def evaluate_relevant_token(self, doc_token_similarity, test_file, evaluate_output):
        fout = open(evaluate_output, "w")
        test_lines = open(test_file).readlines()
        for idx in range(len(doc_token_similarity)):
            line = json.loads(test_lines[idx])
            input_tokens = line["input_tokens"]
            doc_token_sim = [[i, sim] for i, sim in enumerate(doc_token_similarity[idx][:len(input_tokens)])]
            doc_token_sim = sorted(doc_token_sim,  key=lambda x:x[1], reverse=True)
            similar_tokens = [input_tokens[i] for i, sim in doc_token_sim]
            fout.write("   ".join(similar_tokens) + "\t" + " ".join(input_tokens) + "\n")
        fout.close()