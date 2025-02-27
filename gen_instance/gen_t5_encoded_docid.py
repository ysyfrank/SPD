"""
    此脚本用于实现文档的编码，包括树形结构、PQ、url、summarization等
"""
import os
import re
import math
import time
import json
import nanopq
import pickle
import random
import argparse
import collections
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import Counter
from collections import defaultdict
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

parser = argparse.ArgumentParser()
parser.add_argument("--cluster_num", default=256, type=int, help="The number of clusters in each sub-space.")
parser.add_argument("--sub_space", default=24, type=int, help='The number of sub-spaces for 768-dim vector.')
parser.add_argument("--input_path", default="/mnt/xxx/data/msmarco-data/msmarco-docs-sents.top.100k.json", type=str, help='data path')
# parser.add_argument("--input_path", default="/mnt/xxx/data/dual_data/bert_512_doc_top_300w.txt", type=str, help='data path')
parser.add_argument("--output_path", default="/mnt/xxx/data/encoded_docid/t5_atomic_top_100k.txt", type=str, help='output path')
parser.add_argument("--pretrain_model_path", default="/mnt/t5-base", type=str, help='bert model path')
args = parser.parse_args()

def load_doc_vec():
    docid_2_idx = {}
    idx_2_docid = {}
    doc_embeddings = []

    with open(args.input_path, "r") as fr:
        for line in tqdm(fr, desc="loading doc vectors..."):
            did, demb = line.strip().split('\t')
            d_embedding = [float(x) for x in demb.split(',')]

            docid_2_idx[did] = len(docid_2_idx)
            idx_2_docid[docid_2_idx[did]] = did
            
            doc_embeddings.append(d_embedding)

    print("successfully load doc embeddings.")
    return docid_2_idx, idx_2_docid, np.array(doc_embeddings, dtype=np.float32)

## Encoding documents with product quantization
def product_quantization(docid_2_idx, idx_2_docid, doc_embddings):
    print("generating product quantization docids...")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size

    # Instantiate with M sub-spaces, Ks clusters in each sub-space
    pq = nanopq.PQ(M=args.sub_space, Ks=args.cluster_num)

    # Train codewords
    print("training codewords...")
    pq.fit(doc_embeddings)
    print(np.array(pq.codewords).shape)

    with open('/mnt/xxx/data/encoded_docid/codewords_pq24_rand_300k.pkl', 'wb') as fp:
        pickle.dump(pq.codewords, fp)

    # Encode to PQ-codes
    print("encoding doc embeddings...")
    X_code = pq.encode(doc_embeddings)  # [#doc, 8] with dtype=np.uint8

    with open(args.output_path, "w") as fw:
        for idx, doc_code in tqdm(enumerate(X_code), desc="writing doc code into the file..."):
            docid = idx_2_docid[idx]
            new_doc_code = [int(x) for x in doc_code]
            for i, x in enumerate(new_doc_code):
                new_doc_code[i] = int(x) + i*256
            code = ','.join(str(x+vocab_size) for x in new_doc_code)
            fw.write(docid + "\t" + code + "\n")    

## Encoding documents with atomic unique docid
def atomic_unique_docid():
    print("generating atomic docids...")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    encoded_docids = {}

    with open(args.input_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='loading all docids'):
            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)
            
            encoded_docids[docid] = vocab_size + doc_index
    
    with open(args.output_path, "w") as fw:
        for docid, code in encoded_docids.items():
            doc_code = str(code)
            fw.write(docid + "\t" + doc_code + "\n")

## Encoding documents with random docid string (000000 -- 999999), following Google
def naive_string_docid():
    print("generating naive string docids...")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    all_docids = []

    with open(args.input_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='loading all docids'):
            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)
            
            all_docids.append(docid)
    
    code_length = int(math.log(len(all_docids)-1, 10)) + 1
    print(f"#docid: {len(all_docids)}, code_length: {code_length}")
    random.shuffle(all_docids)
    with open(args.output_path, "w") as fw:
        for idx, docid in tqdm(enumerate(all_docids), desc="encoding all documents..."):
            doc_code = [0] * code_length
            temp, index = idx, -1
            while(temp):
                doc_code[index] = temp%10
                temp //= 10
                index -= 1
            doc_code = ','.join([str(x+vocab_size) for x in doc_code])
            fw.write(docid + "\t" + doc_code + "\n")

## Encoding documents with semantically structured identifiers, following Google
def semantic_structured_docid(docid_2_idx, idx_2_docid, doc_embddings): # max 32238
    print("generating semantic structured docids...")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    encoded_docids = {}
    for docid in docid_2_idx:
        encoded_docids[docid] = []
    complete = set()
    max_cluster, max_layer = [0], [0]

    k, c = 10, 100

    def reverse(cur_idx_2_docid, embeddings, layer):
        clusters = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        labels = clusters.labels_
        for i in range(k):
            one_cluster = np.where(labels == i)[0]
            cluster_size = len(one_cluster)
            if layer == 0:
                print(f"processing layer {layer}, cluster {i}, cluster_size {cluster_size}.")
            for index in one_cluster:
                encoded_docids[cur_idx_2_docid[index]].append(i)

            if cluster_size > c and len(set(labels)) > 1:
                temp_embeddings = embeddings[labels == i]
                temp_idx_2_docid = {}
                for index in one_cluster:
                    temp_idx_2_docid[len(temp_idx_2_docid)] = cur_idx_2_docid[index]
                reverse(temp_idx_2_docid, temp_embeddings, layer+1)
            else:
                random.shuffle(one_cluster)
                max_layer[0] = max(max_layer[0], layer)

                if cluster_size > c:
                    print("duplicate embedding: ", cluster_size)
                    total_pos, temp = 1, cluster_size // c
                    while(temp > 0):
                        total_pos += 1
                        temp //= 10
                    print("total position: ", total_pos)
                    
                    for index in range(cluster_size):
                        code = [0] * total_pos
                        code[-1] = index % 100
                        code_idx, temp = -2, index // 100
                        while(temp > 0):
                            code[code_idx] = temp
                            temp //= 10
                            code_idx -= 1

                        encoded_docids[cur_idx_2_docid[one_cluster[index]]].extend(code[:])
                        assert cur_idx_2_docid[one_cluster[index]] not in complete
                        complete.add(cur_idx_2_docid[one_cluster[index]])
                else:
                    for index in range(cluster_size):
                        encoded_docids[cur_idx_2_docid[one_cluster[index]]].append(index)
                        assert cur_idx_2_docid[one_cluster[index]] not in complete
                        complete.add(cur_idx_2_docid[one_cluster[index]])

    reverse(idx_2_docid, doc_embeddings, 0)
    with open(args.output_path, "w") as fw:
        for docid, code in encoded_docids.items():
            max_layer[0] = max(max_layer[0], len(code))
            doc_code = ','.join([str(x+vocab_size) for x in code]+["1"])
            # doc_code = ','.join([str(x) for x in code]+["1"])
            fw.write(docid + "\t" + doc_code + "\n")
    print("max cluster: ", max_cluster, "max layer: ", max_layer)

## Encoding documents with token-id in url
def url_title_docid():
    print("generating url title docids...")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    encoded_docids = {}
    max_length = 0
    max_docid_len = 99
    length_dict = defaultdict(int)

    urls = {}

    with open(args.input_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='loading all docids'):
            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)

            url = doc_item['url'].lower()
            title = doc_item['title'].lower().strip()

            url = url.replace("http://","").replace("https://","").replace("-"," ").replace("_"," ").replace("?"," ").replace("="," ").replace("+"," ").replace(".html","").replace(".php","").replace(".aspx","").strip()
            reversed_url = url.split('/')[::-1]
            url_content = " ".join(reversed_url[:-1])
            domain = reversed_url[-1]
            
            url_content = ''.join([i for i in url_content if not i.isdigit()])
            url_content = re.sub(' +', ' ', url_content).strip()

            if len(title.split()) <= 2:
                url = url_content + " " + domain
            else:
                url = title + " " + domain
            
            encoded_docids[docid] = tokenizer(url).input_ids[:-1][:max_docid_len] + [1]  # max docid length
            #encoded_docids[docid] = tokenizer(url).input_ids[:-1][::-1] + [1] # reverse
            max_length = max(max_length, len(encoded_docids[docid]))
            length_dict[len(encoded_docids[docid])] += 1
    with open(args.output_path, "w") as fw:
        for docid, code in encoded_docids.items():
            doc_code = ','.join([str(x) for x in code])
            fw.write(docid + "\t" + doc_code + "\n")
    print("max length: ", max_length)
    print("length dict: ", sorted(length_dict.items(), key=lambda x:x[0], reverse=True))

if __name__ == "__main__":
    # docid_2_idx, idx_2_docid, doc_embeddings = load_doc_vec()
    # product_quantization(docid_2_idx, idx_2_docid, doc_embeddings)
    atomic_unique_docid()
    # naive_string_docid()
    # semantic_structured_docid(docid_2_idx, idx_2_docid, doc_embeddings)
    # url_structured_docid()