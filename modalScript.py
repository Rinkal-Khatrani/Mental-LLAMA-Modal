from transformers import AutoTokenizer, LlamaForCausalLM

# Install sentencepiece
# You can run this command in your terminal or command prompt.
# pip install sentencepiece

# tokenizer = AutoTokenizer.from_pretrained(
#     "NEU-HAI/mental-alpaca", legacy=False)
# model = AutoModelForCausalLM.from_pretrained("NEU-HAI/mental-alpaca")

model = LlamaForCausalLM.from_pretrained("NEU-HAI/mental-alpaca")
tokenizer = AutoTokenizer.from_pretrained("NEU-HAI/mental-alpaca")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
# generate_ids = model.generate(inputs.input_ids, max_length=30)
# response = tokenizer.batch_decode(
#     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(inputs)
