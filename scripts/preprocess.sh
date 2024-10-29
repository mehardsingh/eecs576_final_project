#!/bin/bash

# Run the Python script with the default arguments explicitly passed
python src/preprocess.py \
    --original_dir "data/original" \
    --purchase_dir "data/purchases" \
    --ordered_csv_fns "2019-Oct.csv" "2019-Nov.csv" "2019-Dec.csv" "2020-Jan.csv" "2020-Feb.csv" "2020-Mar.csv" "2020-Apr.csv" \
    --concat_fp "data/concat.csv" \
    --min_seq_len 5 \
    --user_filter_fp "data/user_filter.csv" \
    --shard_dir "data/shards" \
    --all_shards_fp "data/user_histories.jsonl" \
    --category_filter_fp "data/user_histories_filtered.jsonl" \
    --product2token_fp "data/product2token.json" \
    --split_dir "data/splits" \
    --max_seq_len 50
