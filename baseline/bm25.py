import os
import json
import jieba
import random
import argparse
import collections
from tqdm import tqdm
from evaluate import evaluator
from collections import Counter
from collections import defaultdict
from transformers import BertTokenizer
from gensim.summarization import bm25

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_len", default=128, type=int, help="max sequence length of model. default to 512.")
parser.add_argument("--add_doc_id", default=True, type=bool, help='whether to add the doc id to vocab')
parser.add_argument("--bert_model_path", default="/home/xxx/transformers_models/bert/", type=str, help='bert model path')
parser.add_argument("--doc_file_path", default="/mnt/xxx/data/NQ-data/nq-docs-sents.json", type=str, help='data path')
parser.add_argument("--query_path", default="/mnt/xxx/data/NQ-data/nq-all-queries.tsv", type=str, help='data path')
parser.add_argument("--qrels_path", default="/mnt/xxx/data/NQ-data/nq-dev-qrels.tsv", type=str, help='data path')
parser.add_argument("--output_path", default="/home/xxx/data/train_data/msmarco.add.100k.json", type=str, help='output path')
parser.add_argument("--split_by_tokenizer", default=False, type=bool, help="to split doc/query by space or tokenizer.")

args = parser.parse_args()

def generate_corpus(doc_file_path):
    """
        return: a list of docs (split)
    """
    if args.split_by_tokenizer:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    
    corpus = {}
    with open(args.doc_file_path) as fin:
        for i, line in tqdm(enumerate(fin), desc='constructing all_documents corpus'):
            data = json.loads(line)
            docid = data['docid'].lower()
            title = data['title']
            body = data['body']
            content = title + ' ' + body
            if args.split_by_tokenizer:
                corpus[docid].append(tokenizer.tokenize(content))
            else:
                corpus[docid].append(content.split()[:args.max_seq_len])
    return corpus

def generate_queries():
    all_docid = []
    with open(args.doc_file_path) as fin:
        for i, line in tqdm(enumerate(fin), desc='constructing all_documents list'):
            data = json.loads(line)
            docid = data['docid'].lower()
            all_docid += [docid]

    qid_2_query = {}
    docid_2_qid = defaultdict(list)  # 点击了某个doc的queryid
    valid_queries = {}
    with open(args.query_path) as fin:
        for line in tqdm(fin):
            qid, query = line.strip().split("\t")
            qid_2_query[qid] = query
    
    count = 0
    with open(args.qrels_path) as fin:
        for line in tqdm(fin):
            qid, _, docid, _ = line.strip().split()
            docid = docid.lower()
            if docid not in all_docid:
                continue
            
            docid_2_qid[docid].append(qid)
            count += 1
    print("total count of clicks: ", count)
        
    for docid, qids in tqdm(docid_2_qid.items()):
        for qid in qids:
            query = qid_2_query[qid].lower()
            if args.split_by_tokenizer:
                query_terms = tokenizer.tokenize(query)
            else:
                query_terms = query.split()
            
            if qid in valid_queries:
                valid_queries[qid][1].append(docid)
            else:
                valid_queries[qid] = [query_terms, [docid]]

    return valid_queries

def test_gensim_bm25():
    docid_list =[]
    corpus=[]
    f=open(args.doc_file_path, "r")
    for line in tqdm(f):
        doc = json.loads(line)
        content = doc['body']
        docid = doc['docid'].lower()
        seg_list=content.split()
        corpus.append(seg_list)
        docid_list.append(docid)
    bm25Model = bm25.BM25(corpus)

    test_strs = generate_queries()
    prediction = []
    truth = []
    i=0
    for qid in test_strs.keys():
        i+=1
        query = test_strs[qid][0]
        docid = test_strs[qid][1]
        label = docid_list.index(docid[0])
        scores = bm25Model.get_scores(query)
        prediction.append(scores)
        truth.append(label)
    myevaluator = evaluator()
    result = myevaluator.evaluate_ranking(truth, prediction)
    _mrr100, _mrr, _ndcg100, _ndcg20, _ndcg10, _map20, _p1, _p10, _p20, _p100 = result
    print(f"step {i}, mrr100: {_mrr100}, mrr: {_mrr}, p@1: {_p1}, p@10:{_p10}, p@20: {_p20}, p@100: {_p100}")

if __name__ == '__main__':
    all_doc_file_list = ["nq-docs-sents.json"]
    for doc_file in all_doc_file_list:
        args.doc_file_path = os.path.join("/mnt/xxx/data/NQ-data/", doc_file)
        print("processing doc_file_path: ", args.doc_file_path)
        test_gensim_bm25()