from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "NEU-HAI/mental-alpaca", use_auth_token='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')
model = AutoModelForCausalLM.from_pretrained(
    "NEU-HAI/mental-alpaca", use_auth_token='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')

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=30)
response = tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(response)
