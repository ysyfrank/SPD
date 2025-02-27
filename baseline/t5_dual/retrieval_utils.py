import sys
sys.path += ['./']
import os
import faiss
import argparse
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
parser = argparse.ArgumentParser()
parser.add_argument("--topk", default=100, type=int, help="max sequence length of model. default to 512.")
parser.add_argument("--max_seq_len", default=512, type=int, help="max sequence length of model. default to 512.")
parser.add_argument("--add_doc_id", default=True, type=bool, help='whether to add the doc id to vocab')
parser.add_argument("--bert_model_path", default="/home/xxx/transformers_models/bert/", type=str, help='bert model path')
parser.add_argument("--doc_file_path", default="/home/xxx/data/msmarco-docs-sents.100k.json", type=str, help='data path')
parser.add_argument("--query_path", default="/home/xxx/msmarco-unziped/msmarco-docdev-queries.tsv", type=str, help='data path')
parser.add_argument("--qrels_path", default="/home/xxx/msmarco-unziped/msmarco-docdev-qrels.tsv", type=str, help='data path')
parser.add_argument("--output_path", default="results.txt", type=str, help='output path')
parser.add_argument("--split_by_tokenizer", default=False, type=bool, help="to split doc/query by space or tokenizer.")
args = parser.parse_args()

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

def index_retrieve(index, query_embeddings, topk, batch=None):
    print("Query Num", len(query_embeddings))
    start = timer()
    if batch is None:
        _, nearest_neighbors = index.search(query_embeddings, topk)
    else:
        query_offset_base = 0
        pbar = tqdm(total=len(query_embeddings))
        nearest_neighbors = []
        query_embeddings = query_embeddings.astype(np.float32)
        while query_offset_base < len(query_embeddings):
            batch_query_embeddings = query_embeddings[query_offset_base:query_offset_base+ batch]
            batch_nn = index.search(batch_query_embeddings, topk)[1]
            nearest_neighbors.extend(batch_nn.tolist())
            query_offset_base += len(batch_query_embeddings)
            pbar.update(len(batch_query_embeddings))
        pbar.close()

    elapsed_time = timer() - start
    elapsed_time_per_query = 1000 * elapsed_time / len(query_embeddings)
    print(f"Elapsed Time: {elapsed_time:.1f}s, Elapsed Time per query: {elapsed_time_per_query:.1f}ms")
    return nearest_neighbors

def construct_flatindex_from_embeddings(embeddings, ids=None):
    dim = embeddings.shape[1]
    print('embedding shape: ' + str(embeddings.shape))
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    if ids is not None:
        ids = ids.astype(np.int64)
        embeddings = embeddings.astype(np.float32)
        print(ids.shape, ids.dtype)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(embeddings, ids)
    else:
        index.add(embeddings)
    return index
        
if __name__ == "__main__":
    query_emb_path = "/mnt/xxx/dual_data/t5_512_norepeat_rand_300k.txt"
    doc_embed_path = "/mnt/xxx/data/dual_data/t5_512_doc_rand_300k.txt"
    args.qrels_path = "/mnt/xxx/data/msmarco-data/msmarco-norepeat-qrels.tsv"
    fq=open(query_emb_path,"r")
    fd=open(doc_embed_path,"r")
    qrels_dev = {}
    
    with open(args.qrels_path) as qf:
        for line in tqdm(qf, desc="loading all clicks"):
            qid, _, docid, _ = line.strip().split()
            docid = docid.lower().strip('[d').strip(']')
            if qid in qrels_dev:
                qrels_dev[qid].append(docid)
            else:
                qrels_dev[qid] = [docid]

    
    doc_ids = []
    query_ids = []
    doc_embeddings = []
    query_embeddings = []

    print("load query and doc embeddings...")
    for line in tqdm(fq, desc="loading query embeddings"):
        qid, qemb = line.strip().split('\t')
        q_embedding = qemb.split(',')

        # q_embedding = [float(x) for x in q_embedding]

        # query_ids.append(int(qid.lstrip("q")))
        query_ids.append(qid)
        query_embeddings.append(q_embedding)


    for line in tqdm(fd, desc="loading doc embeddings"):
        did, demb = line.strip().split('\t')
        d_embedding = demb.split(',')

        # d_embedding = [float(x) for x in d_embedding]

        doc_ids.append(int(did.strip('[d').strip(']')))
        doc_embeddings.append(d_embedding)

    doc_ids = np.array(doc_ids)
    # query_ids = np.array(query_ids)
    doc_embeddings = np.array(doc_embeddings)
    query_embeddings = np.array(query_embeddings)

    # print("doc_embddings: ", doc_embeddings.shape, ", query embeddings: ", query_embeddings.shape)
    # print("doc embedddings: ", doc_embeddings[:20])
    # print("query embeddings: ", query_embeddings[:20])
    # match_scores = query_embeddings.dot(doc_embeddings.T) # [q_num, d_num]
    # # print(match_scores[:20, :20])
    # top_index = match_scores.argsort()[:, -args.topk:]
    # nearest_neighbors = []
    # for i in tqdm(range(len(top_index))):
    #     indexes = top_index[i][::-1]
    #     nearest_neighbors.append([])
    #     for index in indexes:
    #         nearest_neighbors[-1].append(doc_ids[index])
    # print("find nearest neighbors.")
    

    print("building index...")
    index = construct_flatindex_from_embeddings(doc_embeddings, doc_ids)
    faiss.omp_set_num_threads(12)
    nearest_neighbors = index_retrieve(index, query_embeddings, args.topk, batch=32)

    mrr_10_list = []
    mrr_100_list = []
    ndcg_10_list = []
    p_1_list = []
    p_10_list = []
    p_20_list = []
    p_100_list = []
    r_1_list = []
    r_10_list = []
    r_20_list = []
    r_100_list = []
    for qid, neighbors in tqdm(zip(query_ids, nearest_neighbors)):
        qid = str(qid)
        score_100 = 0.0
        score_10 = 0.0
        for idx, pid in enumerate(neighbors):
            pid = str(pid)
            if pid in qrels_dev[qid]:
                score_100 = 1.0 / (idx + 1.0)
                if idx < 10:
                    score_10 = 1.0 / (idx + 1.0)
                break
        # print(f"click doc: {qrels_dev[qid]}, ranked doc: {neighbors[:20]}")
        mrr_10_list.append(score_10)
        mrr_100_list.append(score_100)

        neighbors = [str(pid) for pid in neighbors]
        ndcg_10_list.append(NDCG(qrels_dev[qid], neighbors[:10]))

        intersec = len(set(neighbors[:1]) & set(qrels_dev[qid]))
        p_1_list.append(intersec / max(1., float(len(neighbors[:1]))))
        r_1_list.append(intersec)

        intersec = len(set(neighbors[:10]) & set(qrels_dev[qid]))
        p_10_list.append(intersec / max(1., float(len(neighbors[:10]))))
        r_10_list.append(intersec)

        intersec = len(set(neighbors[:20]) & set(qrels_dev[qid]))
        p_20_list.append(intersec / max(1., float(len(neighbors[:20]))))
        r_20_list.append(intersec)

        intersec = len(set(neighbors[:100]) & set(qrels_dev[qid]))
        p_100_list.append(intersec / max(1., float(len(neighbors[:100]))))
        r_100_list.append(intersec)

    print("mrr@100: ", np.mean(mrr_100_list))
    print("mrr@10: ", np.mean(mrr_10_list))
    print("p@1: ", np.mean(p_1_list))
    print("p@10: ", np.mean(p_10_list))
    print("p@20: ", np.mean(p_20_list))
    print("p@100: ", np.mean(p_100_list))
    print("r@1: ", np.mean(r_1_list))
    print("r@10: ", np.mean(r_10_list))
    print("r@20: ", np.mean(r_20_list))
    print("r@100: ", np.mean(r_100_list))