import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Install necessary libraries
# You can run these commands in your terminal or command prompt.
# pip install sentencepiece
# pip install transformers

# Function to run the script

token = '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'


def run_script():
    try:
        logging.debug("Initializing tokenizer and model.")
        # tokenizer = AutoTokenizer.from_pretrained("NEU-HAI/mental-alpaca", legacy=False)
        # model = AutoModelForCausalLM.from_pretrained("NEU-HAI/mental-alpaca")
        tokenizer = AutoTokenizer.from_pretrained(
            "NEU-HAI/Llama-2-7b-alpaca-cleaned", token=token)
        # model = AutoModelForCausalLM.from_pretrained(
        #     "NEU-HAI/Llama-2-7b-alpaca-cleaned", token=token)

        prompt = "Hey, are you conscious? Can you talk to me?"
        # inputs = tokenizer(prompt, return_tensors="pt")
        logging.debug("Generating text.")
        # generate_ids = model.generate(inputs.input_ids, max_length=30)
        # response = tokenizer.batch_decode(
        #     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        logging.debug("Generated response: %s")

    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        raise  # Re-raise the exception for detailed traceback in logs


if __name__ == "__main__":
    try:
        logging.debug("Script started.")
        run_script()
        logging.debug("Script finished.")

    except Exception as e:
        logging.exception("Script terminated with an exception: %s", str(e))
