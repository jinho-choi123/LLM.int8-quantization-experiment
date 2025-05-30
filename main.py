import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
prompt = "The secret to baking a good cake is "

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# text to tokens
token_ids = tokenizer([prompt], return_tensors="pt").to("cuda")

# generate tokens using llm
generated_ids = model.generate(**token_ids, max_length=512)

# print the generated tokens
print(f"{tokenizer.batch_decode(generated_ids)[-1]}")

