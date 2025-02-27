#!/bin/bash

source activate webbrain

## set hyper-parameters in the script file
python /mnt/xxx/gen_instance/gen_t5_encoded_docid.py --input_path /mnt/xxx/data/msmarco-data/msmarco-docs-sents.top.100k.json --output_path /mnt/xxx/data/encoded_docid/t5_atomic_top_100k.txt

python gen_train_data.py

python gen_eval_data.py