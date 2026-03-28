import json
import os

# INPUT: your current small dataset
INPUT_FILE = "finance_dataset.json"

# OUTPUT: new dataset in proper format
OUTPUT_FILE = "finance_dataset_clean.json"

# Load original dataset
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = []

# Convert each entry
for row in data:
    text = row.get("text", "").strip()  # get the existing 'text' field
    if not text:
        continue
    cleaned.append({
        "instruction": text,
        "input": "",
        "output": text
    })

# Save cleaned dataset
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2)

print("Original entries:", len(data))
print("Converted entries:", len(cleaned))
print("Saved to", OUTPUT_FILE)