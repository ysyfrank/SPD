import os

## encoding config  # 319927/321631/231695
atomic_config = {"encoding": "atomic", "add_doc_num":100000, "max_docid_length":1, "use_origin_head": "False"}
semantic_config = {"encoding": "semantic_structured", "add_doc_num":100, "max_docid_length":12, "use_origin_head": "False"}
pq24_config = {"encoding": "pq24", "add_doc_num":6144, "max_docid_length":24, "use_origin_head": "False"}
url_title_config = {"encoding": "url_title", "add_doc_num":6144, "max_docid_length":100, "use_origin_head": "True"}
doc2query_config = {"encoding": "doc2query", "add_doc_num":0, "max_docid_length":100, "use_origin_head": "True"}

config = atomic_config
# config = semantic_config
# config = pq24_config
# config = url_title_config
# config = doc2query_config
encoding, add_doc_num, max_docid_length, use_origin_head = config["encoding"], config["add_doc_num"], config["max_docid_length"], config["use_origin_head"]
# max_docid_length = 80

## training settings
scale = "100k"  # 100k/300k/300w
top_or_rand = "top"  # top/rand
model = "t5_128_10"  # the data for current training
load_model = "t5_128_10"  # the data to be loaded
all_data = "pretrain"  # all data used for training  # pretrain_post_finetune
cur_data = "passage"  # the data used for current training  # pretrain / rank_pretrain / finetune
stage = "pretrain"  # pretrain / post_pretrain / finetune
load_ckpt = "False"  # True if load checkpoint, go to load_ckpt_path
load_gtr = "False"
operation = "training"  # training / pair_training
max_seq_length = 128
save_every_n_epoch = 1

code_dir = "/mnt/xxx"
def main():
    os.system(f"cd {code_dir}/xxx/pretrain && python runT5.py \
        --epoch 20 \
        --per_gpu_batch_size  200 \
        --learning_rate 1e-3 \
        --save_path {code_dir}/xxx/outputs/{model}_{top_or_rand}_{scale}_{encoding}_{all_data}/ \
        --log_path {code_dir}/xxx/logs/{stage}.{model}.{top_or_rand}.{scale}.{encoding}.{all_data}.log \
        --doc_file_path {code_dir}/data/msmarco-data/msmarco-docs-sents.{top_or_rand}.{scale}.json \
        --pretrain_model_path {code_dir}/transformers_models/t5-base \
        --docid_path {code_dir}/data/encoded_docid/t5_{encoding}_{top_or_rand}_{scale}.txt \
        --train_file_path {code_dir}/data/train_data_{top_or_rand}_{scale}/{cur_data}.{model}.{encoding}.{scale}.json \
        --test_file_path {code_dir}/data/test_data_{top_or_rand}_{scale}/ \
        --dataset_script_dir ../data_scripts \
        --dataset_cache_dir ../../negs_tutorial_cache \
        --add_doc_num {add_doc_num} \
        --max_seq_length {max_seq_length} \
        --max_docid_length {max_docid_length} \
        --use_origin_head {use_origin_head} \
        --load_gtr {load_gtr} \
        --load_ckpt {load_ckpt} \
        --load_ckpt_path {code_dir}/xxx/outputs/{load_model}_{top_or_rand}_{scale}_{encoding}_pretrain/model_19.pkl \
        --output_every_n_step 5000 \
        --save_every_n_epoch {save_every_n_epoch} \
        --operation {operation}")

    print("write success")

if __name__ == '__main__':
    main()