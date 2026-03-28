import json

# Path to your dataset
dataset_path = "cleaned_finance_dataset.json"

# Load dataset
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Total samples: {len(dataset)}\n")

# Initialize counters
empty_output = 0
missing_keys = 0
lengths = []

# Final check
for idx, entry in enumerate(dataset):
    # Check for missing keys
    if not all(k in entry for k in ["instruction", "input", "output"]):
        print(f"Missing key in entry {idx}")
        missing_keys += 1
    
    # Check empty output
    if entry.get("output", "").strip() == "":
        print(f"Empty output in entry {idx}")
        empty_output += 1
    
    # Track lengths (simulate tokenization)
    lengths.append(len(entry.get("instruction", "")) + len(entry.get("input", "")) + len(entry.get("output", "")))

print(f"Entries with missing keys: {missing_keys}")
print(f"Entries with empty output: {empty_output}")
print(f"Min length: {min(lengths)}")
print(f"Max length: {max(lengths)}")
print(f"Average length: {sum(lengths)/len(lengths):.2f}")

# Optional: show the first few tokenization previews
print("\nSample tokenization preview (first 3 entries):")
for i in range(min(3, len(dataset))):
    print(f"\nEntry {i}:")
    print("Instruction:", dataset[i]["instruction"])
    print("Input:", dataset[i]["input"])
    print("Output:", dataset[i]["output"])