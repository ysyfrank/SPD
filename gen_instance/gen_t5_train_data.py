import os
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
parser.add_argument("--pretrain_model_path", default="/home/xxx/replearn/transformers_models/bert/", type=str, help='bert model path')
parser.add_argument("--data_path", default="/home/xxx/data/msmarco-docs-sents.100k.json", type=str, help='data path')
parser.add_argument("--docid_path", default=None, type=str, help='docid path')
parser.add_argument("--source_docid_path", default=None, type=str, help='train docid path')
parser.add_argument("--query_path", default="/home/xxx/msmarco-unziped/msmarco-doctrain-queries.tsv", type=str, help='data path')
parser.add_argument("--qrels_path", default="/home/xxx/msmarco-unziped/msmarco-doctrain-qrels.tsv", type=str, help='data path')
parser.add_argument("--output_path", default="/home/xxx/data/train_data/msmarco.add.100k.json", type=str, help='output path')
parser.add_argument("--sample_for_one_doc", default=4, type=int, help="max number of passages sampled for one document.")
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
            docid = data['docid'].lower()
            new_tokens.append("[{}]".format(docid))
    id_to_token = {vocab[k]:k for k in vocab}
    token_to_id = {id_to_token[k]:k for k in id_to_token}
    maxvid = max([k for k in id_to_token])
    start_doc_id = maxvid + 1
    for i, doc_id in enumerate(new_tokens):
        id_to_token[start_doc_id+i] = doc_id
        token_to_id[doc_id] = start_doc_id+i

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

def load_encoded_dict(filename):
    train_dict = {}
    f=open(filename,"r")
    for line in tqdm(f):
        docid, encoded_docid = line.strip().split('\t')
        train_dict[docid] = encoded_docid
    return train_dict

def build_idf(doc_file_path):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    vocab = tokenizer.get_vocab()

    doc_count = 0
    idf_dict = {key: 0 for key in vocab}
    with open(doc_file_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='building term idf dict'):
            doc_count += 1
            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)
            title, url, body = doc_item["title"], doc_item["url"], doc_item["body"]
            all_terms = set(tokenizer.tokenize((title + ' ' + body).lstrip().lower()))

            for term in all_terms:
                if term not in idf_dict:
                    continue
                idf_dict[term] += 1
    
    for key in tqdm(idf_dict):
        idf_dict[key] = np.log(doc_count / (idf_dict[key]+1))

    return idf_dict

# 下面生成各种预训练任务的训练样本
# 任务1.1：MLM任务，[CLS] passage [masked docid] [SEP]
def gen_passage_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    
    sample_count = 0
    fw = open(args.output_path, "w")
    with open(args.data_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='generating samples'):
            max_num_tokens = args.max_seq_length - 1

            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)
            sents_list = doc_item['sents']
            title = doc_item['title'].lower().strip()
            head_terms = tokenizer.tokenize(title)
            current_chunk = head_terms[:]
            current_length = len(head_terms)
            
            sent_id = 0
            sample_for_one_doc = 0
            while sent_id < len(sents_list):
                sent = sents_list[sent_id].lower()
                sent_terms = tokenizer.tokenize(sent)
                current_chunk += sent_terms
                current_length += len(sent_terms)

                if sent_id == len(sents_list) - 1 or current_length >= max_num_tokens: 
                    tokens = current_chunk[:max_num_tokens] + ["</s>"] # truncate the sequence

                    training_instance = {
                        "doc_index":docid,
                        "encoded_docid":encoded_docid[docid],
                        "tokens": tokens,
                    }
                    training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
                    fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
                    sample_count += 1

                    sample_for_one_doc += 1
                    if sample_for_one_doc >= args.sample_for_one_doc:
                        break
                    
                    current_chunk = head_terms[:]
                    current_length = len(head_terms)
                
                sent_id += 1
    fw.close()
    print("total count of samples: ", sample_count)

def gen_sample_terms_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    
    sample_count = 0
    fw = open(args.output_path, "w")
    _, top_or_rand, scale, _ = os.path.split(args.data_path)[1].split(".")

    if os.path.exists(os.path.join(os.path.split(args.output_path)[0], f"sampled_terms.t5_128_1.public.{scale}.json")):
        with open(os.path.join(os.path.split(args.output_path)[0], f"sampled_terms.t5_128_1.public.{scale}.json"), "r") as fr:
            for line in fr:
                line = json.loads(line)
                line["doc_id"] = encoded_docid[line["query_id"]]
                fw.write(json.dumps(line, ensure_ascii=False)+"\n")
                sample_count += 1
    
    else:
        idf_dict = build_idf(args.data_path)
        with open(args.data_path) as fin:
            for doc_index, line in tqdm(enumerate(fin), desc='generating samples'):
                max_num_tokens = args.max_seq_length - 1

                doc_item = json.loads(line)
                docid = doc_item['docid'].lower()
                docid = "[{}]".format(docid)
                title = doc_item['title'].lower().strip()
                body = doc_item['body'].lower().strip()
                all_terms = tokenizer.tokenize(title + ' ' + body)[:1024]
                
                temp_tfidf = []
                all_valid_terms = []
                all_term_tfidf = []
                for term in all_terms:
                    if term not in idf_dict:
                        continue
                    tf_idf = all_terms.count(term) / len(all_terms) * idf_dict[term]
                    temp_tfidf.append((term, tf_idf))
                    all_term_tfidf.append(tf_idf)
                if len(all_term_tfidf) < 10:
                    continue
                tfidf_threshold = sorted(all_term_tfidf, reverse=True)[min(max_num_tokens, len(all_term_tfidf))-1]
                for idx, (term, tf_idf) in enumerate(temp_tfidf):
                    if tf_idf >= tfidf_threshold:
                        all_valid_terms.append(term)

                if len(set(all_valid_terms)) < 2:
                    continue

                tokens = all_valid_terms[:max_num_tokens] + ["</s>"]
                training_instance = {
                    "query_id":docid,
                    "doc_id":encoded_docid[docid],
                    "input_ids": my_convert_tokens_to_ids(tokens, token_to_id),
                }

                fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
                sample_count += 1

    fw.close()
    print("total count of samples: ", sample_count)

# 任务2：MLM任务，[CLS] query [masked docid] [SEP]
def gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
         
    # 从ms-marco数据集中检索出点击了docid的对应query 
    fw = open(args.output_path, "w")
    qid_2_query = {}
    docid_2_qid = defaultdict(list)  # 点击了某个doc的queryid
    with open(args.query_path) as fin:
        for line in tqdm(fin, desc="reading all queries"):
            qid, query = line.strip().split("\t")
            qid_2_query[qid] = query
    
    count = 0
    with open(args.qrels_path) as fin:
        for line in tqdm(fin, desc="reading all click samples"):
            qid, _, docid, _ = line.strip().split()
            
            docid = "[{}]".format(docid.lower())
            if docid not in token_to_id:
                continue
            
            docid_2_qid[docid].append(qid)
            count += 1
    print("total count of clicks: ", count)
    
    max_num_tokens = args.max_seq_length - 1
    
    for docid, qids in tqdm(docid_2_qid.items(), desc="constructing click samples"):
        for qid in qids:
            query = qid_2_query[qid].lower()
            query_terms = tokenizer.tokenize(query)
            tokens = query_terms[:max_num_tokens] + ["</s>"]

            training_instance = {
                "doc_index":docid,
                "encoded_docid":encoded_docid[docid],
                "tokens": tokens,
            }
            training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
            fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
      
    fw.close()

# 任务2：MLM任务，[CLS] query [masked docid] [SEP]
def gen_fake_query_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
         
    # 从ms-marco数据集中检索出点击了docid的对应query 
    fw = open(args.output_path, "w")   
    max_num_tokens = args.max_seq_length - 1
        
    _, top_or_rand, scale, _ = os.path.split(args.data_path)[1].split(".")
    with open(os.path.split(args.query_path)[0] + f"/fake_query_10_all.txt", "r") as fr:
        for line in tqdm(fr, desc="load all fake queries"):
            docid, query = line.strip("\n").split("\t")
            if docid not in token_to_id:
                continue

            query_terms = tokenizer.tokenize(query.lower())
            tokens = query_terms[:max_num_tokens] + ["</s>"]

            training_instance = {
                "doc_index":docid,
                "encoded_docid":encoded_docid[docid],
                "tokens": tokens,
            }
            training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
            fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")

    fw.close()

def gen_fake_finetune_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
         
    # 从ms-marco数据集中检索出点击了docid的对应query 
    fw = open(args.output_path, "w")
    qid_2_query = {}
    docid_2_qid = defaultdict(list)  # 点击了某个doc的queryid
    with open(args.query_path) as fin:
        for line in tqdm(fin, desc="reading all queries"):
            qid, query = line.strip().split("\t")
            qid_2_query[qid] = query
    
    count = 0
    with open(args.qrels_path) as fin:
        for line in tqdm(fin, desc="reading all real click samples"):
            qid, _, docid, _ = line.strip().split()
            
            docid = "[{}]".format(docid.lower())
            if docid not in token_to_id:
                continue
            
            docid_2_qid[docid].append(qid)
            count += 1
    
    max_num_tokens = args.max_seq_length - 1
    
    for docid, qids in tqdm(docid_2_qid.items(), desc="constructing real click samples"):
        for qid in qids:
            query = qid_2_query[qid].lower()
            query_terms = tokenizer.tokenize(query)
            tokens = query_terms[:max_num_tokens] + ["</s>"]

            training_instance = {
                "doc_index":docid,
                "encoded_docid":encoded_docid[docid],
                "tokens": tokens,
            }
            training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
            fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
    
    _, top_or_rand, scale, _ = os.path.split(args.data_path)[1].split(".")
    with open(os.path.split(args.query_path)[0] + f"/fake_query_top_300w.txt", "r") as fr:
        for line in tqdm(fr, desc="load all fake queries"):
            docid, query = line.strip("\n").split("\t")
            if docid not in token_to_id or docid in docid_2_qid:
                continue

            query_terms = tokenizer.tokenize(query.lower())
            tokens = query_terms[:max_num_tokens] + ["</s>"]

            training_instance = {
                "doc_index":docid,
                "encoded_docid":encoded_docid[docid],
                "tokens": tokens,
            }
            training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
            fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")

    fw.close()

# 生成 doc2query
def gen_doc2query_instance(id_to_token, token_to_id, all_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

    # 从ms-marco数据集中检索出点击了docid的对应query 
    fw = open(args.output_path, "w")
    qid_2_query = {}
    did_2_doc = {}
    with open(args.query_path) as fin:
        for line in tqdm(fin, desc="reading all queries"):
            qid, query = line.strip().split("\t")
            qid_2_query[qid] = query.strip()
  
    with open(args.data_path) as fin:
        for line in tqdm(fin, desc="reading all docs"):
            doc_item = json.loads(line)
            docid = doc_item['docid']
            docid = "[{}]".format(docid.lower())
            title = doc_item['title'].lower().strip()
            doc = title + " " + doc_item['body'].lower()
            did_2_doc[docid] = doc.lstrip()
    
    count = 0
    max_num_tokens = args.max_seq_length - 1
    with open(args.qrels_path) as fin:
        for line in tqdm(fin, desc="reading all click samples"):
            qid, _, docid, _ = line.strip().split()
            
            docid = "[{}]".format(docid.lower())
            # if docid not in token_to_id:
            #     continue

            input_tokens = tokenizer.tokenize(did_2_doc[docid])[:max_num_tokens] + ["</s>"]
            output_tokens = tokenizer.tokenize(qid_2_query[qid])[:max_num_tokens] + ["</s>"]
            input_ids = my_convert_tokens_to_ids(input_tokens, token_to_id)
            output_ids = my_convert_tokens_to_ids(output_tokens, token_to_id)

            training_instance = {
                "input_ids": input_ids,
                "query_id": qid,
                "doc_id": ",".join([str(x) for x in output_ids]),
            }
            fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
    fw.close()

def gen_auto_encoder_instance(id_to_token, token_to_id, all_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
         
    # 从ms-marco数据集中检索出点击了docid的对应query 
    fw = open(args.output_path, "w")
    qid_2_query = {}
    docid_2_title = {}
    docid_2_qid = defaultdict(list)  # 点击了某个doc的queryid

    max_num_tokens = args.max_seq_length - 1
    with open(args.query_path) as fin:
        for line in tqdm(fin, desc="reading all queries"):
            qid, query = line.strip().split("\t")
            query_terms = tokenizer.tokenize(query)
            qid_2_query[qid] = query_terms[:max_num_tokens] + ["</s>"]
            
            # input_ids = my_convert_tokens_to_ids(qid_2_query[qid], token_to_id)
            # training_instance = {
            #     "input_ids": input_ids,
            #     "query_id": qid,
            #     "doc_id": ",".join([str(x) for x in input_ids]),
            # }
            # fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")

    with open(args.data_path) as fin:
        for line in tqdm(fin, desc="reading all docs"):
            doc_item = json.loads(line)
            docid = doc_item['docid']
            docid = "[{}]".format(docid.lower())
            title = doc_item['title'].lower().strip()
            
            if len(title.split()) <= 2:
                continue
            
            title_terms = tokenizer.tokenize(title)
            docid_2_title[docid] = title_terms[:max_num_tokens] + ["</s>"]
            input_ids = my_convert_tokens_to_ids(docid_2_title[docid], token_to_id)

            training_instance = {
                "input_ids": input_ids,
                "query_id": docid,
                "doc_id": ",".join([str(x) for x in input_ids]),
            }
            fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
    
    count = 0
    with open(args.qrels_path) as fin:
        for line in tqdm(fin, desc="reading all click samples"):
            qid, _, docid, _ = line.strip().split()      
            docid = "[{}]".format(docid.lower())
            if docid not in token_to_id or docid not in docid_2_title:
                continue
            
            input_ids = my_convert_tokens_to_ids(qid_2_query[qid], token_to_id)
            output_ids = my_convert_tokens_to_ids(docid_2_title[docid], token_to_id)
            training_instance = {
                "input_ids": input_ids,
                "query_id": docid,
                "doc_id": ",".join([str(x) for x in output_ids]),
            }
            fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
            count += 1
    print("total count of clicks: ", count)
    fw.close()

def gen_enhanced_docid_instance(label_filename, train_filename):
    fw = open(args.output_path, "w")
    label_dict = load_encoded_dict(label_filename)
    train_dict = load_encoded_dict(train_filename)
    for docid, encoded in train_dict.items():
        input_ids = [int(item) for item in encoded.split(',')]
        training_instance = {"input_ids": input_ids, "query_id": docid, "doc_id": label_dict[docid]}
        fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
    fw.close()
    

if __name__ == "__main__":
    id_to_token, token_to_id, all_docid, all_term = add_docid_to_vocab(args.data_path)
    
    if args.current_data == "passage":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_passage_instance(id_to_token, token_to_id, all_docid, encoded_docid)

    if args.current_data == "sampled_terms":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_sample_terms_instance(id_to_token, token_to_id, all_docid, encoded_docid)
    
    if args.current_data == "query":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid)
    
    if args.current_data == "fake_query":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_fake_query_instance(id_to_token, token_to_id, all_docid, encoded_docid)

    if args.current_data == "fake_finetune":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_fake_finetune_instance(id_to_token, token_to_id, all_docid, encoded_docid)
    
    if args.current_data == "enhanced_docid":
        gen_enhanced_docid_instance(args.docid_path, args.source_docid_path)
    
    if args.current_data == "doc2query":
        gen_doc2query_instance(id_to_token, token_to_id, all_docid)

    if args.current_data == "auto_encoder":
        gen_auto_encoder_instance(id_to_token, token_to_id, all_docid)