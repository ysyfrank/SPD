import re
import json
import random
import pickle
import argparse
import collections
import numpy as np
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from transformers import T5Tokenizer

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_len", default=512, type=int, help="max sequence length of model. default to 512.")
parser.add_argument("--add_doc_id", default=True, type=bool, help='whether to add the doc id to vocab')
parser.add_argument("--pretrain_model_path", default="/home/xxx/transformers_models/bert/", type=str, help='pretrain model path')
parser.add_argument("--data_path", default="/home/xxx/data/msmarco-docs-sents.json", type=str, help='data path')
parser.add_argument("--query_path", default="/home/xxx/msmarco-unziped/msmarco-docdev-queries.tsv", type=str, help='data path')
parser.add_argument("--qrels_path", default="/home/xxx/msmarco-unziped/msmarco-docdev-qrels.tsv", type=str, help='data path')
parser.add_argument("--output_path", default="/home/xxx/data/twotower.test.300k.json", type=str, help='output path')
parser.add_argument("--query_outpath", default="/home/xxx/data/twotower.query.300k.json", type=str, help='output path')
parser.add_argument("--doc_outpath", default="/home/xxx/data/twotower.doc.300k.json", type=str, help='output path')

args = parser.parse_args()

def my_convert_tokens_to_ids(tokens:list, token_to_id:dict): # token_to_id 是 word:id
    res = []
    for i, t in enumerate(tokens):
        if t in token_to_id:
            res += [token_to_id[t]]
        else:
            res += [token_to_id['<unk>']]
    return res

def my_convert_ids_to_tokens(input_ids:list, id_to_token:dict): # id_to_token 是 id:word
    res = []
    for i, iid in enumerate(input_ids):
        if iid in id_to_token:
            res += [id_to_token[iid]]
        else:
            print("error!")
    return res

def add_padding(training_instance, tokenizer, id_to_token, token_to_id):
    query_ids = my_convert_tokens_to_ids(training_instance['query_tokens'], token_to_id)
    doc_ids = my_convert_tokens_to_ids(training_instance['doc_tokens'], token_to_id)

    new_instance = {
        "query_ids": query_ids,
        "queryid": training_instance["queryid"],
        "doc_ids": doc_ids,
        "docid": training_instance["docid"]
    }
    return new_instance

def add_docid_to_vocab(doc_file_path):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    vocab = tokenizer.get_vocab()
    new_tokens = []
    with open(doc_file_path) as fin:
        for i, line in tqdm(enumerate(fin), desc='constructing all_documents list'):
            data = json.loads(line)
            sents = data['sents']
            docid = data['docid'].lower()

            new_tokens.append("[{}]".format(docid))
    id_to_token = {vocab[k]:k for k in vocab}
    maxvid = max([k for k in id_to_token])
    start_doc_id = maxvid + 1
    for i, doc_id in enumerate(new_tokens):
        id_to_token[start_doc_id + i] = doc_id
    token_to_id = {id_to_token[k]:k for k in id_to_token}
    
    return id_to_token, token_to_id, new_tokens

# 生成 query -- doc 的pair，用于模型训练
def gen_train_instance(id_to_token, token_to_id, all_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

    max_seq_length  = args.max_seq_len

    # Account for [CLS], [SEP], [SEP]
         
    # 从ms-marco数据集中检索出点击了docid的对应query 
    fw = open(args.output_path, "w")
    did_2_doc = {}
    qid_2_query = {}
    docid_2_qid = defaultdict(list)  # 点击了某个doc的queryid
    
    with open(args.data_path) as fin:
        for i, line in tqdm(enumerate(fin), desc='constructing all_documents list'):
            data = json.loads(line)
            title = data['title'].lower()
            body = data['body'].lower()
            docid = data['docid'].lower()
            docid = "[{}]".format(docid)

            if "nq-" in args.data_path:
                pattern = re.compile(r'<[^>]+>',re.S)
                body = pattern.sub('', body)

            did_2_doc[docid] = (title + ' ' + body).lstrip()
    
    with open(args.query_path) as fin:
        for line in tqdm(fin):
            qid, query = line.strip().split("\t")
            qid_2_query[qid] = query
    
    count = 0
    with open(args.qrels_path) as fin:
        for line in tqdm(fin):
            qid, _, docid, _ = line.strip().split()
            docid = "[{}]".format(docid.lower())
            if docid not in token_to_id:
                continue
            
            docid_2_qid[docid].append(qid)
            count += 1
    print("total count of clicks: ", count)
    
    max_num_tokens = max_seq_length - 1
    
    for docid, qids in tqdm(docid_2_qid.items()):
        doc = did_2_doc[docid]
        doc_terms = tokenizer.tokenize(doc)[:max_num_tokens]
        
        doc_tokens = doc_terms + ["</s>"]
        

        for qid in qids:
            query = qid_2_query[qid].lower()
            query_terms = tokenizer.tokenize(query)[:max_num_tokens]

            query_tokens = query_terms + ["</s>"]

            training_instance = {
                "query_tokens": query_tokens,
                "queryid": qid,
                "doc_tokens": doc_tokens,
                "docid": docid,
            }
            training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
            fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
      
    fw.close()

# 生成 query / doc，用于模型做inference
def gen_inference_instance(id_to_token, token_to_id, all_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
         
    # 从ms-marco数据集中检索出点击了docid的对应query 
    fdoc = open(args.doc_outpath, "w")
    fquery = open(args.query_outpath, "w")
    
    did_2_doc = {}
    qid_2_query = {}
    docid_2_qid = defaultdict(list)  # 点击了某个doc的queryid

    max_num_tokens = args.max_seq_len - 1
    
    with open(args.data_path) as fin:
        for i, line in tqdm(enumerate(fin), desc='constructing all_documents list'):
            data = json.loads(line)
            title = data['title'].lower()
            body = data['body'].lower()
            docid = data['docid'].lower()
            docid = "[{}]".format(docid)

            if "nq-" in args.data_path:
                pattern = re.compile(r'<[^>]+>',re.S)
                body = pattern.sub('', body)

            doc_terms = tokenizer.tokenize((title + ' ' + body).lstrip())[:max_num_tokens]
            doc_tokens = doc_terms + ["</s>"]

            training_instance = {
                "query_tokens": doc_tokens,
                "queryid": docid,
                "doc_tokens": doc_tokens[:1],
                "docid": docid,
            }
            training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
            fdoc.write(json.dumps(training_instance, ensure_ascii=False)+"\n")         
    
    with open(args.query_path) as fin:
        for line in tqdm(fin):
            qid, query = line.strip().split("\t")
            qid_2_query[qid] = query
    
    count = 0
    
    with open(args.qrels_path) as fin:
        for line in tqdm(fin):
            qid, _, docid, _ = line.strip().split()
            docid = "[{}]".format(docid.lower())
            if docid not in token_to_id:
                continue
            
            docid_2_qid[docid].append(qid)
            count += 1
    print("total count of clicks: ", count)
    
    
    for docid, qids in tqdm(docid_2_qid.items()):
        for qid in qids:
            query = qid_2_query[qid].lower()
            query_terms = tokenizer.tokenize(query)[:max_num_tokens]

            query_tokens = query_terms + ["</s>"]

            training_instance = {
                "query_tokens": query_tokens,
                "queryid": qid,
                "doc_tokens": query_tokens[:1],
                "docid": qid,
            }
            training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
            fquery.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
      
    fdoc.close()
    fquery.close()


if __name__ == "__main__":
    id_to_token, token_to_id, all_docid = add_docid_to_vocab(args.data_path)

    # gen_train_instance(id_to_token, token_to_id, all_docid)
    gen_inference_instance(id_to_token, token_to_id, all_docid)