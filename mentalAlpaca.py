from transformers import AutoTokenizer, AutoModelForCausalLM

access_token = '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'

tokenizer = AutoTokenizer.from_pretrained(
    "NEU-HAI/Llama-2-7b-alpaca-cleaned", token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    "NEU-HAI/Llama-2-7b-alpaca-cleaned", token=access_token)

prompt = '''Analyze this instruction in an ABA session and tell me what is it Token economy
"Now, so we're going to be using this talking board. And when you reach the goal, then you're going to get the puzzle."'''
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(
    inputs.input_ids, max_length=len(prompt))
response = tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print("Response", response)
