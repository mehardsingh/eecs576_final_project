import json

unique_products = set()
unique_users = set()
unique_sessions = set()
total_purchases = 0

splits = ["data/splits/test.jsonl", "data/splits/train.jsonl", "data/splits/val.jsonl"]

for f in splits:
    with open(f, "r") as file:
        for line in file:
            data = json.loads(line)
            unique_products.update(data["products"])
            unique_users.add(data["user_id"])
            unique_sessions.update(data["sessions"])
            total_purchases += len(data["times"])

print("Number of unique products:", len(unique_products))
print("Number of unique users:", len(unique_users))
print("Number of unique sessions:", len(unique_sessions))
print("Number of total purchases:", total_purchases)