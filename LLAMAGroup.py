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
        # tokenizer = AutoTokenizer.from_pretrained(
        #     "NEU-HAI/mental-alpaca", legacy=False)
        # model = AutoModelForCausalLM.from_pretrained("NEU-HAI/mental-alpaca")
        tokenizer = AutoTokenizer.from_pretrained(
            "klyang/MentaLLaMA-chat-7B", legacy=False)
        model = AutoModelForCausalLM.from_pretrained(
            "klyang/MentaLLaMA-chat-7B", token=token)

        prompt1 = '''Provide a step-by-step guide on teaching a child to greet others appropriately."'''
        prompt2 = '''Describe strategies for encouraging a child with autism to engage in cooperative play with peers.'''

        prompt3 = '''Outline a behavior intervention plan for reducing a child's tantrums during transitions.'''
        prompts = [
            # '''Explain how to reinforce positive behaviors in a child with ADHD using token economies.''',
            '''"Under which key domain or verbal operand this instruction lies Therapist is responding to the child. ""Very nice!""
?"''', '''"Under which key domain or verbal operand this instruction lies Therapist is making the child ready for the session: ""OK, ready?""
?"''', '''"please provide all the ABA Domains in which given conversation lies
Therapist is giving instruction to the child. ""Touch Banana""
Therapist is responding to the child. ""That's right, banana.""
Therapist is praising the child ""Wooo!""
Therapist is responding to the child. ""Very good!"""'''
        ]

        responses = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            logging.debug("Generating text.")
            generate_ids = model.generate(
                inputs.input_ids, max_length=len(prompt))
            response = tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            responses.append(response)
            logging.debug("Generated response: %s", response)

        # Save responses to a file
        with open("responses.txt", "w", encoding="utf-8") as file:
            for prompt, response in zip(prompts, responses):
                file.write(
                    f"Prompt:\n{prompt}\n\nResponse:\n{response}\n\n{'='*50}\n")

        logging.debug("Generated response: %s", response)

    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        raise  # Re-raise the exception for detailed traceback in logs


if __name__ == "__main__":
    try:
        logging.debug("Script started....")
        run_script()
        logging.debug("Script finished.")

    except Exception as e:
        logging.exception("Script terminated with an exception: %s", str(e))
