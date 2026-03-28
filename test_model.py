from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./finance-model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

prompt = "### Instruction:\nWhat should I do during a market crash?\n\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))