import json
import os

FILE_PATH = "finance_dataset_clean.json"  # ✅ your real dataset

# 1️⃣ Check if file exists
if not os.path.exists(FILE_PATH):
    print(f"ERROR: File {FILE_PATH} not found!")
    exit(1)

# 2️⃣ Try to load JSON safely
try:
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    print("ERROR: Failed to load JSON:", e)
    exit(1)

print("\n===== DATASET AUDIT =====\n")

print("Total samples:", len(data))

sample = data[0] if data else {}
print("Sample keys:", list(sample.keys()) if sample else "Empty dataset")

# Missing / empty output
missing_output = 0
empty_output = 0
for row in data:
    if "output" not in row:
        missing_output += 1
    elif not str(row["output"]).strip():
        empty_output += 1

print("Missing output:", missing_output)
print("Empty output:", empty_output)

# Length stats
lengths = [len(str(row.get("output", ""))) for row in data]
if lengths:
    print("Min length:", min(lengths))
    print("Max length:", max(lengths))
    print("Avg length:", sum(lengths)//len(lengths))
else:
    print("No lengths to measure")

# Duplicates
texts = [str(row) for row in data]
duplicates = len(texts) - len(set(texts))
print("Duplicates:", duplicates)

print("\n===== END AUDIT =====\n")