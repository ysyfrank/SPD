import os
import time
import torch
import random
import argparse
from utils import *
from tqdm import tqdm
import torch.nn as nn
from trie import Trie
from evaluate import evaluator
from collections import defaultdict
from gensim.summarization import bm25
from torch.utils.data import DataLoader
from T5ForPretrain import T5ForPretrain
from pretrain_dataset import PretrainDataForT5
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5EncoderModel

# 全局的参数
device = torch.device("cuda:0")
parser = argparse.ArgumentParser()
### training settings
parser.add_argument("--epochs", default=2, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--per_gpu_batch_size", default=25, type=int, help="The batch size.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--warmup_ratio", default=0, type=float, help="The ratio of warmup steps.")
parser.add_argument("--output_every_n_step", default=25, type=int, help="The steps to output training information.")
parser.add_argument("--save_every_n_epoch", default=25, type=int, help="The epochs to save the trained models.")
parser.add_argument("--operation", default="training", type=str, help="which operation to take, training/testing")
parser.add_argument("--use_docid_rank", default="False", type=str, help="whether to use docid for ranking, or only doc code.")
parser.add_argument("--load_ckpt", default="False", type=str, help="whether to load a trained model checkpoint.")
parser.add_argument("--load_gtr", default="False", type=str, help="whether to load the pretrained gtr encoder.")
parser.add_argument("--visualization", default="False", type=str, help="whether to load the pretrained gtr encoder.")

### path to load data and save models
parser.add_argument("--save_path", default="./model/", type=str, help="The path to save trained models.")
parser.add_argument("--log_path", default="./log/", type=str, help="The path to save log.")
parser.add_argument("--doc_file_path", default="/home/xxx/data/msmarco-docs-sents.100k.json", type=str, help='path of origin sent data.')
parser.add_argument("--docid_path", default=None, type=str, help='path of the encoded docid.')
parser.add_argument("--train_file_path", type=str, help="the path/directory of the training file.")
parser.add_argument("--test_file_path", type=str, help="the path/directory of the testing file.")
parser.add_argument("--pretrain_model_path", type=str, help="path of the pretrained model checkpoint")
parser.add_argument("--load_ckpt_path", default="./model/", type=str, help="The path to load ckpt of a trained model.")
parser.add_argument("--dataset_script_dir", type=str, help="The path of dataset script.")
parser.add_argument("--dataset_cache_dir", type=str, help="The path of dataset cache.")

### hyper-parameters to control the model
parser.add_argument("--add_doc_num", type=int, help="the number of docid to be added.")
parser.add_argument("--max_seq_length", type=int, default=512, help="the max length of input sequences.")
parser.add_argument("--max_docid_length", type=int, default=1, help="the max length of docid sequences.")
parser.add_argument("--use_origin_head", default="False", type=str, help="whether to load the lm_head from the pretrained model.")
parser.add_argument("--num_beams", default=10, type=int, help="the number of beams.")

args = parser.parse_args()
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
print("batch_size:", args.batch_size)
print("start a new running with args: ", args)

if args.load_gtr == "True":
    args.log_path = args.log_path.replace("t5", "gtr")

logger = open(args.log_path, "a")
logger.write("\n")
logger.write(f"start a new running with args: {args}\n")
tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

def load_data(file_path):
    """
        function: load data from the file_path
        args: file_path  -- a directory or a specific file
    """
    if os.path.isfile(file_path):
        fns = [file_path]
    else:
        data_dir = file_path
        fns = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)]
    print("file path: ", fns)
    return fns

def load_encoded_docid(docid_path):
    encode_2_docid = {}
    encoded_docids = []
    with open(docid_path, "r") as fr:
        for line in fr:
            docid, encode = line.strip().split("\t")
            docid = docid.lower()
            encode = [int(x) for x in encode.split(",")]
            encoded_docids.append(encode)
            encode = ','.join([str(x) for x in encode])
            if encode not in encode_2_docid:
                encode_2_docid[encode] = [docid]
            else:
                encode_2_docid[encode].append(docid)
    return encoded_docids, encode_2_docid

def load_doc_file(doc_file_path):
    docid2info = {}
    with open(args.doc_file_path, "r") as fr:
        for line in tqdm(fr, desc="reading all docs"):
            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)
            title = doc_item['title'].lower().strip()
            url = doc_item['url'].lower()
            body = doc_item['body'].lower()
            docid2info[docid] = [url, title, (title + ' ' + body).lstrip()]
    return docid2info

def load_bert_vector(test_file_path):
    _, _, top_or_rand, scale = os.path.split(test_file_path)[0].split("/")[-1].split("_")
    doc_vec_path = f"/mnt/xxx/data/dual_data/t5_512_doc_{top_or_rand}_{scale}.txt"
    query_vec_path = f"/mnt/xxx/data/dual_data/t5_512_query_{top_or_rand}_{scale}.txt"
    queryid_2_vec, docid_2_vec, query_2_id = {}, {}, {}

    with open(query_vec_path, "r") as fr:
        for line in tqdm(fr, desc="loading query embeddings"):
            qid, qemb = line.strip().split('\t')
            q_embedding = [float(x) for x in qemb.split(',')]
            queryid_2_vec[qid.lstrip("q")] = q_embedding
    
    with open(doc_vec_path, "r") as fr:
        for line in tqdm(fr, desc="loading doc embeddings"):
            did, demb = line.strip().split('\t')
            d_embedding = [float(x) for x in demb.split(',')]
            # did = did.lower().lstrip("[d").rstrip("]")
            docid_2_vec[did] = d_embedding

    query_path = "/mnt/xxx/data/msmarco-data/msmarco-docdev-queries.tsv"
    if scale in ["320k"]:
        query_path = "/mnt/xxx/data/nq-data/nq-docdev-queries.tsv"

    with open(query_path) as fin:
        for line in tqdm(fin, desc="loading query text"):
            qid, query = line.strip().split("\t")
            query_2_id[query] = qid.lstrip("q")
    
    return queryid_2_vec, docid_2_vec, query_2_id
    
def train_model(train_data):
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)

    # expand doc id
    pretrain_model.resize_token_embeddings(pretrain_model.config.vocab_size + args.add_doc_num)
    
    if args.load_gtr == "True":
        print("load parameters of encoder from gtr.")
        encoder_model = T5EncoderModel.from_pretrained(f"/mnt/xxx/transformers_models/gtr-{os.path.split(args.pretrain_model_path)[1]}")
        encoder_model.resize_token_embeddings(encoder_model.config.vocab_size + args.add_doc_num)

        pretrain_state = pretrain_model.state_dict()
        encoder_state = encoder_model.state_dict()
        for k, v in encoder_state.items():
            pretrain_state[k] = v
        pretrain_model.load_state_dict(pretrain_state)

        args.save_path = args.save_path.replace("t5", "gtr")
        args.load_ckpt_path = args.load_ckpt_path.replace("t5", "gtr")

    model = T5ForPretrain(pretrain_model, args)
    
    if args.load_ckpt == "True": # 基于之前的checkpoint开始训练
        save_model = load_model(os.path.join(args.load_ckpt_path))
        model.load_state_dict(save_model)
        print("Successfully load checkpoint from ", args.load_ckpt_path)
    
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data) # 开始训练

def train_step(model, train_data):
    with torch.no_grad():
        for key in train_data.keys():
            if key in ["query_id", "doc_id"]:
                continue
            train_data[key] = train_data[key].to(device)
    input_ids = train_data["input_ids"]
    attention_mask = train_data["attention_mask"]
    labels = train_data["docid_labels"]

    loss = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return loss

def fit(model, X_train):
    print("start training...")
    train_dataset = PretrainDataForT5(X_train, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args) # 构建训练集
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    # optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_ratio * int(t_total), num_training_steps=t_total)
    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        avg_loss = 0
        model.train()
        for i, training_data in enumerate(tqdm(train_dataloader)):
            loss = train_step(model, training_data) # 过模型, 取loss
            loss = loss.mean()
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step() # 更新模型参数
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            if i % args.output_every_n_step == 0:
                localtime = time.asctime(time.localtime(time.time()))
                print(f"{localtime} step: {i}, lr: {args.learning_rate}, loss: {loss.item()}")
                logger.write(f"{localtime} step: {i}, lr: {args.learning_rate}, loss: {loss.item()}\n")
                logger.flush()
            avg_loss += loss.item()
        cnt = len(train_dataset) // args.batch_size + 1
        print("Average loss:{:.6f} ".format(avg_loss / cnt))
        logger.write("Average loss:{:.6f} \n".format(avg_loss / cnt))

        if (epoch+1) % args.save_every_n_epoch == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f"model_{epoch}.pkl"))
            print(f"Save the model in {args.save_path}")
    logger.close()

def evaluate_beamsearch():
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    pretrain_model.resize_token_embeddings(pretrain_model.config.vocab_size + args.add_doc_num)
    model = T5ForPretrain(pretrain_model, args)
    if args.load_gtr == "True":
        args.save_path = args.save_path.replace("t5", "gtr")
    save_model = load_model(args.save_path)
    model.load_state_dict(save_model)
    model = model.to(device)
    model.eval()
    myevaluator = evaluator()

    encoded_docid, encode_2_docid = load_encoded_docid(args.docid_path)
    docid_trie = Trie([[0] + item for item in encoded_docid])

    use_docid_rank = args.use_docid_rank
    rerank_method = "random"
    if args.use_docid_rank == "True":
        if rerank_method == "BM25":
            docid2info = load_doc_file(args.doc_file_path)
        elif rerank_method == "BERT":
            queryid_2_vec, docid_2_vec, query_2_id = load_bert_vector(args.test_file_path)

    def prefix_allowed_tokens_fn(batch_id, sent):
        return docid_trie.get(sent.tolist())

    def docid2string(docid):
        x_list = []
        for x in docid:
            if x != 0:
                x_list.append(str(x))
            if x == 1:
                break
        return ",".join(x_list)

    if os.path.exists(args.test_file_path):
        localtime = time.asctime(time.localtime(time.time()))
        print(f"Evaluate on the {args.test_file_path}.")
        logger.write(f"{localtime} Evaluate on the {args.test_file_path}.\n")
        test_data = load_data(args.test_file_path)  # 加载测试数据集
        test_dataset = PretrainDataForT5(test_data, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args) # 构建训练集
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        truth, prediction, inputs = [], [], []

        for i, testing_data in tqdm(enumerate(test_dataloader)):
            with torch.no_grad():
                for key in testing_data.keys(): # [input_ids, token_type_ids, attention_mask, mlm_labels, label]
                    if key in ["query_id", "doc_id"]:
                        continue
                    testing_data[key] = testing_data[key].to(device)
            
            input_ids = testing_data["input_ids"]
            attention_mask = testing_data["attention_mask"]

            if use_docid_rank == "False":
                labels = testing_data["docid_labels"] # encode
                truth.extend([[docid2string(docid)] for docid in labels.cpu().numpy().tolist()])
            else:
                labels = testing_data["query_id"] # docid
                truth.extend([[docid] for docid in labels])
            
            inputs.extend(input_ids)

            outputs = model.generate(input_ids, max_length=args.max_docid_length+1, num_return_sequences=args.num_beams, num_beams=args.num_beams, do_sample=False, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)

            for j in range(input_ids.shape[0]):
                query_term = tokenizer.decode(input_ids[j], skip_special_tokens=True).split()
                doc_rank = []
                batch_output = outputs[j*args.num_beams:(j+1)*args.num_beams].cpu().numpy().tolist()
                for docid in batch_output:
                    if use_docid_rank == "False":
                        doc_rank.append(docid2string(docid))
                    else:
                        docid_list = encode_2_docid[docid2string(docid)]
                        if len(docid_list) > 1:
                            if rerank_method == "random":
                                random.shuffle(docid_list)
                                doc_rank.extend(docid_list)
                            elif rerank_method == "BM25":
                                corpus = []
                                for docid in docid_list:
                                    corpus.append(docid2info[docid][2].split())
                                bm25Model = bm25.BM25(corpus)
                                scores = bm25Model.get_scores(query_term)

                                bm25_docid_list = [docid_list[index] for index, value in sorted(list(enumerate(scores)), key=lambda x:x[1], reverse=True)]
                                doc_rank.extend(bm25_docid_list)
                            elif rerank_method == "BERT":
                                query_embedding = np.array(queryid_2_vec[query_2_id[' '.join(query_term)]]).reshape(1, -1)
                                doc_embedding = np.array([docid_2_vec[x] for x in docid_list])
                                match_scores = query_embedding.dot(doc_embedding.T)  # [q_num, d_num]
                                
                                bert_docid_list = [docid_list[index] for index, value in sorted(list(enumerate(match_scores)), key=lambda x:x[1], reverse=True)]
                                doc_rank.extend(bert_docid_list)

                        else:
                            doc_rank.extend(docid_list)
                                                    
                prediction.append(doc_rank)

        if args.visualization == "True":
            docid2info = load_doc_file(args.doc_file_path)
            fw = open("visualization_pq.txt","w")
            for idx in range(len(prediction)):
                pred = prediction[idx][0]
                tru = truth[idx][0]
                input_ids = inputs[idx]
                if args.use_docid_rank == "False":
                    pred = encode_2_docid[pred][0]
                    tru = encode_2_docid[tru][0]
                input_query = tokenizer.decode(input_ids, skip_special_tokens=True)
                fw.write("query: " + input_query + " pred_url: " + str(docid2info[pred][0]) + " pred_title: " + str(docid2info[pred][1]) + " label_url: " + str(docid2info[tru][0]) + " label_title: " + str(docid2info[tru][1])+'\n')
                # fw.write("query: " + input_query + " pred_encoding: " + str(pred) + '\n')
                # fw.write("query: " + input_query + " labe_encoding: " + str(tru) + '\n')
                # fw.write("-----------------" + '\n')

        result = myevaluator.evaluate_ranking(truth, prediction, operation="sorted_generate_doc")
        _mrr10, _mrr, _ndcg10, _ndcg20, _ndcg100, _map20, _p1, _p10, _p20, _p100, _r1, _r10, _r100, _r1000 = result
        localtime = time.asctime(time.localtime(time.time()))
        print(f"mrr@10:{_mrr10}, mrr:{_mrr}, p@1:{_p1}, p@10:{_p10}, p@20:{_p20}, p@100:{_p100}, r@1:{_r1}, r@10:{_r10}, r@100:{_r100}, r@1000:{_r1000}")
        logger.write(f"mrr@10:{_mrr10}, mrr:{_mrr}, p@1:{_p1}, p@10:{_p10}, p@20:{_p20}, p@100:{_p100}, r@1:{_r1}, r@10:{_r10}, r@100:{_r100}, r@1000:{_r1000}\n")

def generate():
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    pretrain_model.resize_token_embeddings(pretrain_model.config.vocab_size + args.add_doc_num)
    model = T5ForPretrain(pretrain_model, args)
    save_model = load_model(args.save_path)
    model.load_state_dict(save_model)
    model = model.to(device)
    model.eval()
    myevaluator = evaluator()

    if os.path.exists(args.test_file_path):
        localtime = time.asctime(time.localtime(time.time()))
        print(f"Generate query for the {args.test_file_path}.")
        logger.write(f"{localtime} Generate query for the {args.test_file_path}.\n")
        test_data = load_data(args.test_file_path)  # 加载测试数据集
        test_dataset = PretrainDataForT5(test_data, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args) # 构建训练集
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        
        fw = open(os.path.split(args.test_file_path)[0] + "/gen_query_10_top_320k.txt", "w")

        for i, testing_data in tqdm(enumerate(test_dataloader)):
            with torch.no_grad():
                for key in testing_data.keys(): # [input_ids, token_type_ids, attention_mask, mlm_labels, label]
                    if key in ["query_id", "doc_id"]:
                        continue
                    testing_data[key] = testing_data[key].to(device)
            
            input_ids = testing_data["input_ids"]
            attention_mask = testing_data["attention_mask"]

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False, num_return_sequences=10, num_beams=10)
            decode_query = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for j in range(len(testing_data["query_id"])):
                docid = testing_data["query_id"][j]
                batch_query = decode_query[j*10:(j+1)*10]
                for query in batch_query:
                    fw.write(docid + "\t" + query + "\n")
                    fw.flush()
        fw.close()

# 计算每一步生成之后，候选文档空间的大小。
def count_candidate(generate_step=6):
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    pretrain_model.resize_token_embeddings(pretrain_model.config.vocab_size + args.add_doc_num)
    model = T5ForPretrain(pretrain_model, args)
    if args.load_gtr == "True":
        args.save_path = args.save_path.replace("t5", "gtr")
    save_model = load_model(args.save_path)
    model.load_state_dict(save_model)
    model = model.to(device)
    model.eval()
    myevaluator = evaluator()

    encoded_docid, encode_2_docid = load_encoded_docid(args.docid_path)
    docid_trie = Trie([[0] + item for item in encoded_docid])

    use_docid_rank = args.use_docid_rank

    def prefix_allowed_tokens_fn(batch_id, sent):
        return docid_trie.get(sent.tolist())

    def docid2string(docid):
        x_list = []
        for x in docid:
            if x != 0:
                x_list.append(str(x))
            if x == 1:
                break
        return ",".join(x_list)

    def docidwithsuffix(suffix_len):
        suffix_to_docid = defaultdict(list)
        for encode, docid in encode_2_docid.items():
            encode_suffix = ','.join(encode.split(",")[:suffix_len])
            if use_docid_rank == "False":
                suffix_to_docid[encode_suffix].append(encode)
            else:
                suffix_to_docid[encode_suffix].extend(docid)
        return suffix_to_docid

    suffix_to_docid = docidwithsuffix(generate_step)

    if os.path.exists(args.test_file_path):
        localtime = time.asctime(time.localtime(time.time()))
        print(f"Evaluate on the {args.test_file_path}.")
        logger.write(f"{localtime} Evaluate on the {args.test_file_path}.\n")
        test_data = load_data(args.test_file_path)  # 加载测试数据集
        test_dataset = PretrainDataForT5(test_data, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args) # 构建训练集
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        truth, prediction, inputs = [], [], []

        for i, testing_data in tqdm(enumerate(test_dataloader)):
            with torch.no_grad():
                for key in testing_data.keys(): # [input_ids, token_type_ids, attention_mask, mlm_labels, label]
                    if key in ["query_id", "doc_id"]:
                        continue
                    testing_data[key] = testing_data[key].to(device)
            
            input_ids = testing_data["input_ids"]
            attention_mask = testing_data["attention_mask"]

            if use_docid_rank == "False":
                labels = testing_data["docid_labels"] # encode
                truth.extend([[docid2string(docid)] for docid in labels.cpu().numpy().tolist()])
            else:
                labels = testing_data["query_id"] # docid
                truth.extend([[docid] for docid in labels])
            
            inputs.extend(input_ids)

            outputs = model.generate(input_ids, max_length=generate_step+1, num_return_sequences=args.num_beams, num_beams=args.num_beams, do_sample=False, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)

            for j in range(args.batch_size):
                doc_rank = []
                batch_output = outputs[j*args.num_beams:(j+1)*args.num_beams].cpu().numpy().tolist()
                for docid in batch_output:
                    doc_rank.extend(suffix_to_docid[docid2string(docid)])

                print("candidate count: ", len(doc_rank))                        
                prediction.append(doc_rank)

        result = myevaluator.evaluate_ranking(truth, prediction, operation="sorted_generate_doc")
        _mrr10, _mrr, _ndcg10, _ndcg20, _ndcg100, _map20, _p1, _p10, _p20, _p100, _r1, _r10, _r100, _r1000 = result
        localtime = time.asctime(time.localtime(time.time()))
        print(f"mrr@10:{_mrr10}, mrr:{_mrr}, p@1:{_p1}, p@10:{_p10}, p@20:{_p20}, p@100:{_p100}, r@1:{_r1}, r@10:{_r10}, r@100:{_r100}, r@1000:{_r1000}")
        logger.write(f"mrr@10:{_mrr10}, mrr:{_mrr}, p@1:{_p1}, p@10:{_p10}, p@20:{_p20}, p@100:{_p100}, r@1:{_r1}, r@10:{_r10}, r@100:{_r100}, r@1000:{_r1000}\n")

if __name__ == '__main__':
    if args.operation == "training":
        train_data = load_data(args.train_file_path) # 加载数据集
        set_seed() # 控制各种随机种子
        train_model(train_data) # 开始预训练

    if args.operation == "pair_training":
        train_data = load_data(args.train_file_path) # 加载数据集
        set_seed() # 控制各种随机种子
        train_model_pairwise(train_data) # 开始预训练
    
    if args.operation == "testing":
        evaluate_beamsearch()

    if args.operation == "count_candidate":
        count_candidate(generate_step=6)

    if args.operation == "generation":
        generate()