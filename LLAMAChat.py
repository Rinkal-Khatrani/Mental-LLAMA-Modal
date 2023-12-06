from transformers import AutoTokenizer, AutoModelForCausalLM

access_token = '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'

tokenizer = AutoTokenizer.from_pretrained(
    "klyang/MentaLLaMA-chat-7B", legacy=False)
model = AutoModelForCausalLM.from_pretrained(
    "klyang/MentaLLaMA-chat-7B", token=access_token)

prompt = '''Analyze this instruction in an ABA session and tell me what is it Token economy
"Now, so we're going to be using this talking board. And when you reach the goal, then you're going to get the puzzle."'''
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=len(prompt))
aa = tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Response", aa)
