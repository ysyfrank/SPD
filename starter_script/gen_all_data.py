import os
import json
from tqdm import tqdm

model = "t5"
encoding = "url_title"
source_docid = "url_title" # label_source_docid
add_doc_num = 0
max_seq_length = 128
sample_for_one_doc = 1
scale = "300w"
top_or_rand = "top"
msmarco_or_nq = "msmarco"

def main():
    code_dir = "/mnt/xxx"
    # for stage_id, cur_data in enumerate(["passage", "sampled_terms", "auto_encoder", "fake_query", "enhanced_docid", "query"]):
    #     print(f"generating {cur_data} data...")
    #     sample_for_one_doc = [10,1,1,10,1,1][stage_id]
    #     os.system(f"cd {code_dir}/xxx/webbrain/gen_instance/ && python gen_{model}_train_data.py \
    #         --max_seq_length {max_seq_length} \
    #         --pretrain_model_path {code_dir}/transformers_models/{model}-base \
    #         --data_path {code_dir}/data/{msmarco_or_nq}-data/{msmarco_or_nq}-docs-sents.{top_or_rand}.{scale}.json \
    #         --docid_path {code_dir}/data/encoded_docid/{model}_{encoding}_{top_or_rand}_{scale}.txt \
    #         --source_docid_path {code_dir}/data/encoded_docid/{model}_{source_docid}_{top_or_rand}_{scale}.txt \
    #         --query_path {code_dir}/data/{msmarco_or_nq}-data/{msmarco_or_nq}-doctrain-queries.tsv \
    #         --qrels_path {code_dir}/data/{msmarco_or_nq}-data/{msmarco_or_nq}-doctrain-qrels.tsv \
    #         --top100_path {code_dir}/data/{msmarco_or_nq}-data/{msmarco_or_nq}-doctrain-top100 \
    #         --output_path {code_dir}/data/train_data_{top_or_rand}_{scale}/{cur_data}.{model}_{max_seq_length}_{sample_for_one_doc}.{encoding}.{scale}.json \
    #         --masked_docid_prob 0.2 \
    #         --max_prediction_per_seq 5 \
    #         --add_doc_num {add_doc_num} \
    #         --sample_for_one_doc {sample_for_one_doc} \
    #         --current_data {cur_data}")
    
    # print("merge data for the pretrain stage...")
    # passage_input = f"{code_dir}/data/train_data_{top_or_rand}_{scale}/passage.{model}_{max_seq_length}_5.{encoding}.{scale}.json"
    # sampled_input = f"{code_dir}/data/train_data_{top_or_rand}_{scale}/sampled_terms.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
    # # autoencoder_input = f"{code_dir}/data/train_data_{top_or_rand}_{scale}/auto_encoder.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
    # docid_input = f"{code_dir}/data/train_data_{top_or_rand}_{scale}/enhanced_docid.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
    # merge_output = f"{code_dir}/data/train_data_{top_or_rand}_{scale}/pretrain.{model}_{max_seq_length}_5.{encoding}.{scale}.json"
    # fout = open(merge_output, "w")
    # total_count = 0
    # with open(passage_input, "r") as fr:
    #     for line in tqdm(fr, desc="loading passage input"):
    #         fout.write(line)
    #         total_count += 1
    # with open(sampled_input, "r") as fr:
    #     for line in tqdm(fr, desc="loading sampled terms input"):
    #         fout.write(line)
    #         total_count += 1
    # # with open(autoencoder_input, "r") as fr:
    # #     for line in tqdm(fr, desc="loading autoencoder input"):
    # #         fout.write(line)
    # #         total_count += 1
    # with open(docid_input, "r") as fr:
    #     for line in tqdm(fr, desc="loading docid input"):
    #         fout.write(line)
    #         total_count += 1
    # fout.close()
    # print("total number of pretrain samples: ", total_count)

    print("merge data for the post finetune stage...")
    fakequery_input = f"{code_dir}/data/train_data_{top_or_rand}_{scale}/fake_query.{model}_{max_seq_length}_10.{encoding}.{scale}.json"
    query_input = f"{code_dir}/data/train_data_{top_or_rand}_{scale}/query.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
    merge_output = f"{code_dir}/data/train_data_{top_or_rand}_{scale}/post_finetune.{model}_{max_seq_length}_10.{encoding}.{scale}.json"
    fout = open(merge_output, "w")
    total_count = 0
    with open(fakequery_input, "r") as fr:
        for line in tqdm(fr, desc="loading fakequery input"):
            fout.write(line)
            total_count += 1
    with open(query_input, "r") as fr:
        for line in tqdm(fr, desc="loading query input"):
            fout.write(line)
            total_count += 1
    fout.close()
    print("total number of post finetune samples: ", total_count)

    # print("merge data for the finetune stage...")
    # query_input = f"{code_dir}/data/train_data_{top_or_rand}_{scale}/query.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
    # merge_output = f"{code_dir}/data/train_data_{top_or_rand}_{scale}/finetune.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
    # fout = open(merge_output, "w")
    # total_count = 0
    # with open(query_input, "r") as fr:
    #     for line in tqdm(fr, desc="loading query input"):
    #         fout.write(line)
    #         total_count += 1
    # fout.close()
    # print("total number of finetune samples: ", total_count)
    
    # print("generating evaluate data...")
    # os.system(f"cd {code_dir}/xxx/webbrain/gen_instance/ && python gen_{model}_eval_data.py \
    #     --max_seq_length {max_seq_length} \
    #     --pretrain_model_path {code_dir}/transformers_models/{model}-base \
    #     --data_path {code_dir}/data/{msmarco_or_nq}-data/{msmarco_or_nq}-docs-sents.{top_or_rand}.{scale}.json \
    #     --docid_path {code_dir}/data/encoded_docid/{model}_{encoding}_{top_or_rand}_{scale}.txt \
    #     --query_path {code_dir}/data/{msmarco_or_nq}-data/{msmarco_or_nq}-docdev-queries.tsv \
    #     --qrels_path {code_dir}/data/{msmarco_or_nq}-data/{msmarco_or_nq}-docdev-qrels.tsv \
    #     --top100_path {code_dir}/data/{msmarco_or_nq}-data/{msmarco_or_nq}-docdev-top100 \
    #     --output_path {code_dir}/data/test_data_{top_or_rand}_{scale}/query_dev.{model}_{max_seq_length}_1.{encoding}.{scale}.json \
    #     --add_doc_num {add_doc_num} \
    #     --current_data query_dev")
    
    print("write success")

if __name__ == '__main__':
    main()