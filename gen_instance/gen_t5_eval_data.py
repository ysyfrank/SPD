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
parser.add_argument("--max_seq_length", default=512, type=int, help="max sequence length of model. default to 512.")
parser.add_argument("--add_doc_num", default=0, type=int, help='the number of new docid added to the vocab')
parser.add_argument("--pretrain_model_path", default="/mnt/xxx/replearn/transformers_models/bert/", type=str, help='pretrain model path')
parser.add_argument("--data_path", default="/mnt/xxx/data/msmarco-docs-sents.100k.json", type=str, help='data path')
parser.add_argument("--docid_path", default=None, type=str, help='docid path')
parser.add_argument("--query_path", default="/mnt/xxx/data/msmarco-doctrain-queries.tsv", type=str, help='data path')
parser.add_argument("--qrels_path", default="/mnt/xxx/data/msmarco-unziped/msmarco-doctrain-qrels.tsv", type=str, help='data path')
parser.add_argument("--output_path", default="/mnt/xxx/data/train_data/msmarco.add.100k.json", type=str, help='output path')
parser.add_argument("--current_data", default=None, type=str, help="current generating data.")

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
    input_ids = my_convert_tokens_to_ids(training_instance['tokens'], token_to_id)

    new_instance = {
        "input_ids": input_ids,
        "query_id": training_instance["doc_index"],
        "doc_id": training_instance["encoded_docid"],
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
    token_to_id = {id_to_token[k]:k for k in id_to_token}
    for i, doc_id in enumerate(new_tokens):
        token_to_id[doc_id] = i

    return id_to_token, token_to_id, new_tokens, list(vocab.values())

def get_encoded_docid(docid_path, all_docid, token_to_id):
    encoded_docid = {}
    if docid_path is None:
        for i, doc_id in enumerate(all_docid):
            encoded_docid[doc_id] = str(token_to_id[doc_id])
    else:
        with open(docid_path, "r") as fr:
            for line in fr:
                docid, encode = line.strip().split("\t")
                docid = "[{}]".format(docid.lower().strip('[').strip(']'))
                encoded_docid[docid] = encode
    return encoded_docid

# 下面生成各种预训练任务的训练样本
# 任务1：基于passage来预测masked docid是什么
def gen_passage_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

    sample_count = 0
    # Account for [CLS], [SEP], [SEP]

    doc_file_path = args.data_path
    fw = open(args.output_path, "w")
    with open(doc_file_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='generating samples'):
            max_num_tokens = args.max_seq_length - 1

            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)
            sents_list = doc_item['sents']
            title = doc_item['title'].lower()
            head_terms = tokenizer.tokenize(title)
            current_chunk = head_terms[:]
            current_length = len(head_terms)
            
            sent_id = 0
            while sent_id < len(sents_list):
                sent = sents_list[sent_id].lower()
                sent_terms = tokenizer.tokenize(sent)
                current_chunk += sent_terms
                current_length += len(sent_terms)

                if sent_id == len(sents_list) - 1 or current_length >= max_num_tokens:
                    current_chunk = current_chunk[:max_num_tokens]  # truncate the sequence
                    tokens = current_chunk[:] + ["</s>"]

                    training_instance = {
                        "doc_index":docid,
                        "encoded_docid":encoded_docid[docid],
                        "tokens": tokens,
                    }
                    training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
                    fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
                    sample_count += 1
                    
                    break
                
                sent_id += 1
    fw.close()
    print("total count of samples: ", sample_count)

# 任务2：基于query来预测masked docid是什么
def gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

    qid_2_query = {}
    docid_2_qid = defaultdict(list)  # 点击了某个doc的queryid
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

    max_num_tokens = args.max_seq_length - 1
    fw = open(args.output_path, "w")

    for docid, qids in tqdm(docid_2_qid.items()):
        for qid in qids:
            query = qid_2_query[qid].lower()
            query_terms = tokenizer.tokenize(query)
            tokens = query_terms[:max_num_tokens] + ["</s>"]

            evaluation_instance = {
                "doc_index":docid,
                "encoded_docid":encoded_docid[docid],
                "tokens": tokens,
            }
            evaluation_instance = add_padding(evaluation_instance, tokenizer, id_to_token, token_to_id)
            fw.write(json.dumps(evaluation_instance, ensure_ascii=False)+"\n")
       
    fw.close()

# 任务3：分别生成docid在训练集中的query和docid不在训练集中的query测试
def gen_split_query_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

    qid_2_query = {}
    docid_2_qid = defaultdict(list)  # 点击了某个doc的queryid
    train_docid = {}
    train_qrels_path = args.qrels_path.replace("docdev", "doctrain")

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

    with open(train_qrels_path) as fin:
        for line in tqdm(fin):
            qid, _, docid, _ = line.strip().split()
            docid = "[{}]".format(docid.lower())
            if docid not in docid_2_qid:
                continue

            train_docid[docid] = 1

    max_num_tokens = args.max_seq_length - 1
    
    f_repeat = open(args.output_path.replace("dev", "repeat"), "w")
    f_norepeat = open(args.output_path.replace("dev", "norepeat"), "w")

    for docid, qids in tqdm(docid_2_qid.items()):
        for qid in qids:
            query = qid_2_query[qid].lower()
            query_terms = tokenizer.tokenize(query)
            tokens = query_terms[:max_num_tokens] + ["</s>"]

            evaluation_instance = {
                "doc_index":docid,
                "encoded_docid":encoded_docid[docid],
                "tokens": tokens,
            }
            evaluation_instance = add_padding(evaluation_instance, tokenizer, id_to_token, token_to_id)

            if docid in train_docid:
                f_repeat.write(json.dumps(evaluation_instance, ensure_ascii=False)+"\n")
            else:
                f_norepeat.write(json.dumps(evaluation_instance, ensure_ascii=False)+"\n")
       
    f_repeat.close()
    f_norepeat.close()

# 生成doc 2 query数据，给没有点击query的doc生成query
def gen_doc2query_instance(id_to_token, token_to_id, all_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
         
    # 从ms-marco数据集中检索出点击了docid的对应query 
    fw = open(args.output_path, "w")
        
    max_num_tokens = args.max_seq_length - 1
    with open(args.data_path) as fin:
        for line in tqdm(fin, desc="reading all docs"):
            doc_item = json.loads(line)
            docid = doc_item['docid']
            docid = "[{}]".format(docid.lower())
            title = doc_item['title'].lower().strip()
            doc = title + " " + doc_item['body'].lower()

            input_tokens = tokenizer.tokenize(doc)[:max_num_tokens] + ["</s>"]
            output_tokens = ["</s>"]
            input_ids = my_convert_tokens_to_ids(input_tokens, token_to_id)
            output_ids = my_convert_tokens_to_ids(output_tokens, token_to_id)

            training_instance = {
                "input_ids": input_ids,
                "query_id": docid,
                "doc_id": ",".join([str(x) for x in output_ids]),
            }
            fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
    fw.close()

if __name__ == "__main__":
    id_to_token, token_to_id, all_docid, all_term = add_docid_to_vocab(args.data_path)

    if args.current_data == "passage":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_passage_instance(id_to_token, token_to_id, all_docid, encoded_docid)
    
    if args.current_data == "query_dev":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid)

    if args.current_data == "query_split":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_split_query_instance(id_to_token, token_to_id, all_docid, encoded_docid)

    if args.current_data == "doc2query":
        gen_doc2query_instance(id_to_token, token_to_id, all_docid)