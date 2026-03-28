from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="finance_dataset_v2.json")

def tokenize(example):
    texts = []

    for i in range(len(example["instruction"])):
        text = f"### Instruction:\n{example['instruction'][i]}\n\n"

        if example["input"][i]:
            text += f"### Input:\n{example['input'][i]}\n\n"

        text += f"### Response:\n{example['output'][i]}"
        texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()

model.save_pretrained("./finance-model")
tokenizer.save_pretrained("./finance-model")