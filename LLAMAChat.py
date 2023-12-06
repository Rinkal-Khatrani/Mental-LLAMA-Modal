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
        encoded_input = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoded_input)
        logits = outputs.logits

        mask_token_index = (encoded_input.input_ids == tokenizer.mask_token_id)[
            0].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        aa = tokenizer.decode(predicted_token_id)

        print("Result: ", aa)

    # Ask if the user wants to continue
    user_response = input("Do you want to continue? (yes/no): ").lower()
    if user_response != 'yes':
        break
