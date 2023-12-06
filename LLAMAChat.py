from transformers import AutoTokenizer, AutoModelForCausalLM

access_token = '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'

tokenizer = AutoTokenizer.from_pretrained(
    "klyang/MentaLLaMA-chat-7B", token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    "klyang/MentaLLaMA-chat-7B", token=access_token)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
aa = tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Response", aa)
