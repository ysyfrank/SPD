import os

## encoding config  # 319927/321631/231695   # 9/8//7
atomic_config = {"encoding": "atomic", "add_doc_num":100000, "max_docid_length":1, "use_origin_head": "False"}
semantic_config = {"encoding": "semantic_structured", "add_doc_num":100, "max_docid_length":12, "use_origin_head": "False"}
pq24_config = {"encoding": "pq24", "add_doc_num":6144, "max_docid_length":24, "use_origin_head": "False"}
url_title_config = {"encoding": "url_title", "add_doc_num":6144, "max_docid_length":100, "use_origin_head": "True"}
doc2query_config = {"encoding": "doc2query", "add_doc_num":0, "max_docid_length":100, "use_origin_head": "True"}

config = atomic_config
# config = semantic_config
# config = url_title_config
# config = pq24_config
encoding, add_doc_num, max_docid_length, use_origin_head = config["encoding"], config["add_doc_num"], config["max_docid_length"], config["use_origin_head"]
# add_doc_num = 6144

## test settings
scale = "100k"   # 100k/300k/300w
top_or_rand = "top"  # top/rand
model = "t5_128_1"  # the data for current training
load_model = "t5_128_1"  # the data to be loaded
all_data = "passage"  # all data used for training
cur_data = "query_dev"  # the data used for current training
stage = "inference"  # pretrain / finetune
num_beams = 100
use_docid_rank = "True"  # True to discriminate different docs with the same docid
load_gtr = "False"
operation = "testing"
max_seq_length = 64

code_dir = "/mnt/xxx/"
def main():
    for epoch in [0]: #[1,3,5,7,9,11,13,15,17,19]:
        os.system(f"cd {code_dir}/xxx/webbrain/pretrain && python runT5.py \
            --epoch 10 \
            --per_gpu_batch_size 12 \
            --learning_rate 5e-4 \
            --save_path {code_dir}/xxx/outputs/{load_model}_{top_or_rand}_{scale}_{encoding}_{all_data}/model_{epoch}.pkl \
            --log_path {code_dir}/xxx/logs/{stage}.{model}.{top_or_rand}.{scale}.{encoding}.{all_data}.log \
            --doc_file_path {code_dir}/data/nq-data/nq-docs-sents.{top_or_rand}.{scale}.json \
            --pretrain_model_path {code_dir}/transformers_models/t5-base \
            --docid_path {code_dir}/data/encoded_docid/t5_{encoding}_{top_or_rand}_{scale}.txt \
            --train_file_path {code_dir}/data/train_data_{top_or_rand}_{scale}/{cur_data}.{model}.{encoding}.{scale}.json \
            --test_file_path {code_dir}/data/test_data_{top_or_rand}_{scale}/{cur_data}.{model}.{encoding}.{scale}.json \
            --dataset_script_dir ../data_scripts \
            --dataset_cache_dir ../../negs_tutorial_cache \
            --num_beams {num_beams} \
            --add_doc_num {add_doc_num} \
            --max_seq_length {max_seq_length} \
            --max_docid_length {max_docid_length} \
            --output_every_n_step 1000 \
            --save_every_n_epoch 2 \
            --load_gtr {load_gtr} \
            --operation {operation} \
            --use_docid_rank {use_docid_rank}")

    print("write success")

if __name__ == '__main__':
    main()