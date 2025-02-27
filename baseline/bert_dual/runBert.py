import os
import time
import torch
import pickle
import argparse
from utils import *
from tqdm import tqdm
from BertTwoTower import BertTwoTower
from point_dataset import PointDataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel

# 全局的参数
device = torch.device("cuda:0")
parser = argparse.ArgumentParser()
parser.add_argument("--per_gpu_batch_size", default=25, type=int, help="The batch size.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--epochs", default=2, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--doc_embed_path", default="../data/docs_embed_init.pkl", type=str, help="The path of initialized doc embed.")
parser.add_argument("--save_path", default="./model/", type=str, help="The path to save model.")
parser.add_argument("--log_path", default="./log/", type=str, help="The path to save log.")
parser.add_argument("--doc_file_path", default="/home/xxx/data/msmarco-docs-sents.100k.json", type=str, help='path of origin sent data.')
parser.add_argument("--train_file_path", type=str, help="the path/directory of the training file.")
parser.add_argument("--test_file_path", type=str, help="the path/directory of the testing file.")
parser.add_argument("--pretrain_model_path", type=str, help="path of the bert model checkpoint")
parser.add_argument("--load_ckpt_path", default="./model/", type=str, help="The path to load ckpt model.")
parser.add_argument("--dataset_script_dir", type=str, help="The path to save log.")
parser.add_argument("--dataset_cache_dir", type=str, help="The path to save log.")
parser.add_argument("--add_doc_num", type=int, help="the number of docid to be added.")
parser.add_argument("--max_seq_len", type=int, default=512, help="the max length of input sequences.")
parser.add_argument("--operation", default="training", type=str, help="which operation to take, training/testing")
parser.add_argument("--infer_output", type=str, help="path to store the evaluation output.")
parser.add_argument("--load_ckpt", default=False, type=bool, help="whether to load a trained model checkpoint.")

args = parser.parse_args()
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
print("start a new running with args: ", args)
logger = open(args.log_path, "a+")
logger.write("\n")
logger.write(f"running with args: {args}\n")
tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
vocab_size = tokenizer.vocab_size

def load_data(train_file, test_file):
    if os.path.isfile(train_file):
        train_data = [train_file]
    else:
        data_dir = train_file
        train_data = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)]
    test_data = test_file
    print("train file path: ", train_data, ", test file path: ", test_data)
    return train_data, test_data

def train_model(train_data, test_data):
    bert_model = BertModel.from_pretrained(args.pretrain_model_path)
    model = BertTwoTower(bert_model)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data, test_data)

def train_step(model, train_data):
    with torch.no_grad():
        for key in train_data.keys():
            if key in ["queryid", "docid"]:
                continue
            train_data[key] = train_data[key].to(device)
    loss, _, _ = model.forward(train_data)
    return loss

def fit(model, X_train, X_test):
    train_dataset = PointDataset(X_train, args.max_seq_len, tokenizer, args.dataset_script_dir, args.dataset_cache_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * int(t_total), num_training_steps=t_total)
    one_epoch_step = len(train_dataset) // args.batch_size
    os.makedirs(args.save_path, exist_ok=True)
    best_result = 0.0

    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        avg_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader)
        for i, training_data in enumerate(epoch_iterator):
            loss = train_step(model, training_data)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            # epoch_iterator.set_postfix(lr=args.learning_rate, loss=loss.item())
            if i % 1000 == 0:
                localtime = time.asctime(time.localtime(time.time()))
                print(f"{localtime} step: {i}, lr: {args.learning_rate}, loss: {loss.item()}")
                logger.write(f"{localtime} step: {i}, lr: {args.learning_rate}, loss: {loss.item()}\n")
                logger.flush()
            avg_loss += loss.item()

        cnt = len(train_dataset) // args.batch_size + 1
        logger.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        accuracy = evaluate(model, X_test)
        print("accuracy: ", accuracy)
        if accuracy > best_result:
            best_result = accuracy
            torch.save(model.state_dict(), os.path.join(args.save_path, f"model_{epoch}.pkl"))
        logger.write(f"accuracy: {accuracy}, best_result: {best_result}")
    logger.close()

def evaluate(model, X_test, is_test=False):
    model.eval()
    test_dataset = PointDataset(X_test, args.max_seq_len, tokenizer, args.dataset_script_dir, args.dataset_cache_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    total_preds = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in test_data.keys():
                    if key in ["queryid", "docid"]:
                        continue
                    test_data[key] = test_data[key].to(device)
            loss, y_pred, _ = model.forward(test_data) # [bs, bs]
            total_preds.extend(y_pred.data.cpu().numpy())
    accuracy = np.mean(total_preds)
    return accuracy

def inference(test_data):
    bert_model = BertModel.from_pretrained(args.pretrain_model_path)

    model = BertTwoTower(bert_model)
    model_state_dict = torch.load(args.save_path)
    model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})

    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    test_dataset = PointDataset(test_data, args.max_seq_len, tokenizer, args.dataset_script_dir, args.dataset_cache_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    id_to_embed = {}

    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in test_data.keys():
                    if key in ["queryid", "docid"]:
                        continue
                    test_data[key] = test_data[key].to(device)
            loss, y_pred, embeds = model.forward(test_data) # [bs, bs]
            embeds = embeds.data.cpu()
            for idx, queryid in enumerate(test_data["queryid"]):
                id_to_embed[queryid] = embeds[idx]

    with open(args.infer_output, "w") as fw:
        for k, v in id_to_embed.items():
            v = [str(float(i)) for i in v.numpy().tolist()]
            fw.write(k+'\t'+','.join(v)+'\n')


if __name__ == '__main__':
    train_data, test_data = load_data(args.train_file_path, args.test_file_path)
    set_seed()
    if args.operation == "training":
        train_model(train_data, test_data)
    if args.operation == "inference":
        inference(test_data)