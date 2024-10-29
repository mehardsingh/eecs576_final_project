import pandas as pd
from tqdm import tqdm
import os
from collections import Counter
import json
import numpy as np
import random
import copy

def filter_chunk_event_type(chunk, value="purchase"):
    filtered_chunk = chunk[chunk["event_type"] == value]
    return filtered_chunk

def filter_event_type(data_fp, output_fp, chunksize=100000):
    total_rows = sum(1 for _ in open(data_fp)) - 1
    total_chunks = total_rows // chunksize + (total_rows % chunksize > 0)

    # Use tqdm to wrap the iterable and specify the total number of chunks
    with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        for i, chunk in enumerate(pd.read_csv(data_fp, chunksize=chunksize)):
            filtered_chunk = filter_chunk_event_type(chunk)

            if i == 0:
                filtered_chunk.to_csv(output_fp, index=False, mode='w', header=True)
            else:
                filtered_chunk.to_csv(output_fp, index=False, mode='a', header=False)

            pbar.update(1)

def filter_all_event_type(data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    inp_file_names = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    all_files = [os.path.join(data_dir, f) for f in inp_file_names]
    
    for i, filepath in enumerate(all_files):
        print(f">> filtering {filepath}")
        output_file_path = os.path.join(save_dir, inp_file_names[i])
        filter_event_type(filepath, output_file_path, chunksize=100000)

def concat_csvs(ordered_csv_fps, output_file, chunksize=100000):
    # Initialize an empty DataFrame for writing
    first_file = True  # Flag to check if it's the first file for writing headers

    # Loop through each CSV file
    for file in ordered_csv_fps:
        file_path = os.path.join(file)
        
        # Read the current CSV file in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunksize):  # Adjust chunksize as needed
            # Write the chunk to the output file
            if first_file:
                chunk.to_csv(output_file, mode='w', index=False, header=True)
                first_file = False  # After the first write, set the flag to False
            else:
                chunk.to_csv(output_file, mode='a', index=False, header=False)  # Append without header

    print(f"All files have been concatenated into {output_file}.")

def get_user_freq(csv_file_path, chunksize=100000):
    value_counter = Counter()

    # Get the total number of rows in the CSV file for progress calculation
    total_rows = sum(1 for _ in open(csv_file_path)) - 1  # Subtract 1 for the header
    total_chunks = total_rows // chunksize + (total_rows % chunksize > 0)  # Calculate total chunks

    # Read the CSV file in chunks with a progress bar
    with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        for chunk in pd.read_csv(csv_file_path, chunksize=chunksize):
            # Update the counter with the values from the specified column
            value_counter.update(chunk["user_id"].dropna())  # Drop NaN values if present
            pbar.update(1)  # Update the progress bar after processing each chunk

    return dict(value_counter)

def filter_chunk_user(chunk, valid_users):
    filtered_chunk = chunk[chunk["user_id"].isin(valid_users)]
    return filtered_chunk

def filter_user(csv_file_path, output_fp, user_freq, chunksize=100000, thresh=5):
    valid_users = {user_id for user_id in user_freq if user_freq[user_id] >= thresh}

    total_rows = sum(1 for _ in open(csv_file_path)) - 1
    total_chunks = total_rows // chunksize + (total_rows % chunksize > 0)

    # Use tqdm to wrap the iterable and specify the total number of chunks
    with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        for i, chunk in enumerate(pd.read_csv(csv_file_path, chunksize=chunksize)):
            filtered_chunk = filter_chunk_user(chunk, valid_users)
            filtered_chunk = filtered_chunk[["user_id", "event_time", "category_id", "product_id", "user_session"]]

            if i == 0:
                filtered_chunk.to_csv(output_fp, index=False, mode='w', header=True)
            else:
                filtered_chunk.to_csv(output_fp, index=False, mode='a', header=False)

            pbar.update(1)

def get_unique_users(csv_file_path, chunksize=100000):
    unique_elements = set()
    for chunk in pd.read_csv(csv_file_path, chunksize=chunksize):
        unique_elements.update(chunk['user_id'].unique())  # Replace with your column name

    return list(unique_elements)

def save_user_histories(csv_file_path, save_dir, unique_users, users_per_shard=10000, chunksize=100000):
    os.makedirs(save_dir, exist_ok=True)
    num_shards = (len(unique_users) // users_per_shard) if (len(unique_users) // users_per_shard) == 0 else 1 + (len(unique_users) // users_per_shard)
    print(f"Number of shards: {num_shards}")

    for shard_idx in range(num_shards):
        shard_users = unique_users[int(shard_idx * users_per_shard):int(shard_idx * users_per_shard) + users_per_shard]

        # Get total number of rows in the CSV file (excluding the header)
        total_rows = sum(1 for _ in open(csv_file_path)) - 1

        # Calculate total number of chunks
        total_chunks = total_rows // chunksize + (total_rows % chunksize > 0)

        user_history = {}

        # Use tqdm to add a progress bar for chunk processing
        for chunk in tqdm(pd.read_csv(csv_file_path, chunksize=chunksize), total=total_chunks, desc=f'Processing shard {shard_idx}'):
            shard_user_df = chunk[chunk["user_id"].isin(shard_users)]

            for user in shard_users:
                str_user = str(user)
                if str_user not in user_history:
                    user_history[str_user] = {
                        "times": list(),
                        "products": list(),
                        "sessions": list(),
                        "categories": list()
                    }

                user_df = shard_user_df[shard_user_df["user_id"] == user]
                times = user_df["event_time"].tolist()
                products = user_df["product_id"].tolist()
                sessions = user_df["user_session"].tolist()
                categories = user_df["category_id"].tolist()

                user_history[str_user]["times"] += times
                user_history[str_user]["products"] += products
                user_history[str_user]["sessions"] += sessions
                user_history[str_user]["categories"] += categories

        jsonl_list = []
        for user in user_history:
            user_info = user_history[user]
            user_info["user_id"] = user
            jsonl_list.append(user_info)

        # Path to the jsonl file
        with open(os.path.join(save_dir, f"shard_{shard_idx}.jsonl"), 'w') as f:
            for entry in jsonl_list:
                f.write(json.dumps(entry) + '\n')

def concat_shards(shard_dir, output_fp):
    inp_file_names = [f for f in os.listdir(shard_dir) if os.path.isfile(os.path.join(shard_dir, f))]
    all_files = [os.path.join(shard_dir, f) for f in inp_file_names]

    with open(output_fp, 'w') as outfile:
        for input_file in all_files:
            with open(input_file, 'r') as infile:
                for line in infile:
                    outfile.write(line)

def count_items_categories(user_history_fp):
    item_counter = dict()
    category_counter = Counter()

    item_counter = dict()
    with open(user_history_fp, 'r') as f:
        for line in f:
            data = json.loads(line)
            products = data["products"]
            categories = data["categories"]


            for i in range(len(products)):
                if not products[i] in item_counter:
                    item_counter[products[i]] = {"count": 0, "category": categories[i]}
                item_counter[products[i]]["count"] += 1

            category_counter.update(categories)

    item_counter = dict(item_counter)
    category_counter = dict(category_counter)

    return item_counter, category_counter

def get_top_categories(category_counter, percentile=98):
    purchases_per_category = [pair[1] for pair in category_counter.items()]
    percentile = np.percentile(purchases_per_category, percentile)
    filtered_category_counter = {pair[0]: pair[1] for pair in category_counter.items() if pair[1] > percentile}
    filtered_categories = set(filtered_category_counter.keys())
    return filtered_categories

def filtered_user_histories(inp_fp, out_fp, filtered_categories, user_thresh=5):
    with open(inp_fp, 'r') as inp_f:
        with open(out_fp, 'w') as out_f:
            for line in inp_f:
                data = json.loads(line)
                out_data = {"times": list(), "products": list(), "sessions": list(), "categories": list(), "user_id": data["user_id"]}

                for i in range(len(data["times"])):
                    curr_category = data["categories"][i]
                    if curr_category in filtered_categories:
                        out_data["times"].append(data["times"][i])
                        out_data["products"].append(data["products"][i])
                        out_data["sessions"].append(data["sessions"][i])
                        out_data["categories"].append(data["categories"][i])

                if len(out_data["times"]) >= user_thresh:
                    out_f.write(json.dumps(out_data) + '\n')

def save_product2token(category_filter_fp, product2token_fp):
    product2token = {-2: 0, -1: 1}
    vocab_count = len(product2token)
    with open(category_filter_fp, 'r') as inp_f:
        for line in inp_f:
            data = json.loads(line)
            products = data["products"]
            for product in products:
                if not product in product2token:
                    product2token[product] = vocab_count
                    vocab_count += 1

    with open(product2token_fp, mode="w") as f:
        json.dump(product2token, f, indent=4)

    return product2token

def get_train_val_test(category_filter_fp, split_dir, min_seq_len=5, max_seq_len=50):
    os.makedirs(split_dir, exist_ok=True)
    split_names = ["train", "val", "test"]

    # create split files
    for i in range(len(split_names)):
        with open(os.path.join(split_dir, f"{split_names[i]}.jsonl"), mode="w") as f:
            f.write('')

    with open(category_filter_fp, 'r') as inp_f:
        for line in inp_f:
            data = json.loads(line)

            seq_len = len(data["times"])

            train_val_test_end_idxs = [random.randint(min_seq_len, seq_len-2), seq_len-1, seq_len]
            for i in range(len(split_names)):
                end_idx = train_val_test_end_idxs[i]
                start_idx = max(0, end_idx-max_seq_len)
                data_copy = copy.deepcopy(data)

                for k in data_copy:
                    if not k == "user_id":
                        data_copy[k] = data_copy[k][start_idx:end_idx]

                with open(os.path.join(split_dir, f"{split_names[i]}.jsonl"), mode="a") as f:
                    f.write(json.dumps(data_copy) + '\n')

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Script for processing purchase data.")

    parser.add_argument("--original_dir", type=str, default="data/original", help="Directory for the original data files.")
    parser.add_argument("--purchase_dir", type=str, default="data/purchases", help="Directory for the purchase data files.")
    parser.add_argument("--ordered_csv_fns", type=str, nargs='+', default=[
        "2019-Oct.csv", "2019-Nov.csv", "2019-Dec.csv", "2020-Jan.csv", "2020-Feb.csv", "2020-Mar.csv", "2020-Apr.csv"
    ], help="List of ordered CSV filenames.")
    parser.add_argument("--concat_fp", type=str, default="data/concat.csv", help="Filepath for the concatenated CSV.")
    parser.add_argument("--min_seq_len", type=int, default=5, help="Minimum sequence length.")
    parser.add_argument("--user_filter_fp", type=str, default="data/user_filter.csv", help="Filepath for filtered user data.")
    parser.add_argument("--shard_dir", type=str, default="data/shards", help="Directory for data shards.")
    parser.add_argument("--all_shards_fp", type=str, default="data/user_histories.jsonl", help="Filepath for all user history shards.")
    parser.add_argument("--category_filter_fp", type=str, default="data/user_histories_filtered.jsonl", help="Filepath for category-filtered user histories.")
    parser.add_argument("--product2token_fp", type=str, default="data/product2token.json", help="Filepath for product-to-token mapping.")
    parser.add_argument("--split_dir", type=str, default="data/splits", help="Directory for data splits.")
    parser.add_argument("--max_seq_len", type=int, default=50, help="Maximum sequence length.")

    args = parser.parse_args()
    args.user_thresh = args.min_seq_len + 2
    return args

def main(args):
    # Step 1: Filter events
    filter_all_event_type(args.original_dir, args.purchase_dir)

    # Step 2: Concatenate ordered CSV files
    ordered_purchase_fps = [os.path.join(args.purchase_dir, csv_fn) for csv_fn in args.ordered_csv_fns]
    concat_csvs(ordered_purchase_fps, args.concat_fp)

    # Step 3: Get user frequency
    user_freq = get_user_freq(args.concat_fp)

    # Step 4: Filter users based on frequency threshold
    filter_user(args.concat_fp, args.user_filter_fp, user_freq, thresh=args.user_thresh)

    # Step 5: Get unique users and save their histories in shards
    unique_users = get_unique_users(args.user_filter_fp)
    save_user_histories(args.user_filter_fp, args.shard_dir, unique_users, users_per_shard=10000, chunksize=100000)

    # Step 6: Concatenate shards into a single file
    concat_shards(args.shard_dir, args.all_shards_fp)

    # Step 7: Count items and categories, and filter top categories
    item_counter, category_counter = count_items_categories(args.all_shards_fp)
    filtered_categories = get_top_categories(category_counter, percentile=98)
    filtered_user_histories(args.all_shards_fp, args.category_filter_fp, filtered_categories, user_thresh=args.user_thresh)

    # Step 8: Create a product-to-token mapping and save it
    product2token = save_product2token(args.category_filter_fp, args.product2token_fp)

    # Step 9: Split data into train, validation, and test sets
    get_train_val_test(args.category_filter_fp, args.split_dir, min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len)


if __name__ == "__main__":
    args = parse_args()
    main(args)