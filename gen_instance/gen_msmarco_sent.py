import nltk
import json
from tqdm import tqdm
from collections import defaultdict

doc_file_path = "/home/xxx/msmarco-unziped/msmarco-docs.tsv"
qrels_train_path = "/home/xxx/msmarco-unziped/msmarco-doctrain-qrels.tsv"
qrels_dev_path = "/home/xxx/msmarco-unziped/msmarco-docdev-qrels.tsv"
fout = open("/home/xxx/data/msmarco-docs-sent-unique.json", "w")
id_to_content = {}
doc_click_count = defaultdict(int)
content_to_id = {}

with open(doc_file_path) as fin:
    for i, line in tqdm(enumerate(fin)):
        cols = line.split("\t")
        if len(cols) != 4:
            continue
        docid, url, title, body = cols
        sents = nltk.sent_tokenize(body)
        id_to_content[docid] = {"docid": docid, "url": url, "title": title, "body": body, "sents": sents}
        doc_click_count[docid] = 0

print("Total number of unique documents: ", len(doc_click_count))

with open(qrels_train_path, "r") as fr:
    for line in tqdm(fr):
        queryid, _, docid, _ = line.strip().split()
        doc_click_count[docid] += 1

# 所有doc按照点击query的数量(popularity)由高到低选择，优先使用点击次数多的doc  
sorted_click_count = sorted(doc_click_count.items(), key=lambda x:x[1], reverse=True)
print("sorted_click_count: ", sorted_click_count[:100])
for docid, count in doc_click_count:
    if docid not in id_to_content:
        continue
    fout.write(json.dumps(id_to_content[docid])+"\n")

fout.close()