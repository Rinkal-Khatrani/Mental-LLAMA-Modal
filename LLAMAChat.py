from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

access_token = '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'

tokenizer = AutoTokenizer.from_pretrained(
    "klyang/MentaLLaMA-chat-7B", legacy=False)
model = AutoModelForCausalLM.from_pretrained(
    "klyang/MentaLLaMA-chat-7B", token=access_token)

while True:
    # Take user input
    input_text = input("You: ")

    if len(input_text) > 0:
        # Tokenize the input
        inputs = tokenizer(input_text, return_tensors="pt")
        generate_ids = model.generate(
            inputs.input_ids, max_length=len(input_text))
        generated_text = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Print the result
        print("Bot:", generated_text)

    # Ask if the user wants to continue
    user_response = input("Do you want to continue? (yes/no): ").lower()
    if user_response != 'yes':
        break
