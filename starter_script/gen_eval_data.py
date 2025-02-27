import os

model = "t5"
encoding = "atomic"
add_doc_num = 100000
max_seq_length = 128
sample_for_one_doc = 1
scale = "100k"
cur_data = "query_dev"
top_or_rand = "top"
msmarco_or_nq = "msmarco"

def main():
    code_dir = "/mnt/xxx"
    os.system(f"cd {code_dir}/xxx/gen_instance/ && python gen_{model}_eval_data.py \
        --max_seq_length {max_seq_length} \
        --pretrain_model_path {code_dir}/transformers_models/{model}-base \
        --data_path {code_dir}/data/{msmarco_or_nq}-data/{msmarco_or_nq}-docs-sents.{top_or_rand}.{scale}.json \
        --docid_path {code_dir}/data/encoded_docid/{model}_{encoding}_{top_or_rand}_{scale}.txt \
        --query_path {code_dir}/data/{msmarco_or_nq}-data/{msmarco_or_nq}-docdev-queries.tsv \
        --qrels_path {code_dir}/data/{msmarco_or_nq}-data/{msmarco_or_nq}-docdev-qrels.tsv \
        --output_path {code_dir}/data/test_data_{top_or_rand}_{scale}/query_dev.{model}_{max_seq_length}_{sample_for_one_doc}.{encoding}.{scale}.json \
        --add_doc_num {add_doc_num} \
        --current_data {cur_data}")

    print("write success")

if __name__ == '__main__':
    main()