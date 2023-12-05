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
            "NEU-HAI/Llama-2-7b-alpaca-cleaned", token=token)
        model = AutoModelForCausalLM.from_pretrained(
            "NEU-HAI/Llama-2-7b-alpaca-cleaned", token=token)

        prompt1 = '''Analyze this instruction in an ABA session and tell me what is it Token economy
"Now, so we're going to be using this talking board. And when you reach the goal, then you're going to get the puzzle."'''
        prompt2 = '''For an ABA session where a child is undergoing therapy with a therapist , analyze the conversation  sequentially  between child and therapist and act as a Board Certified Behavior Analyst and do the following. 
1. Take only those conversations where therapist is giving instruction to the child and Identify which verbal operants are present in those instructions from the list below. 
2. For each conversation set identify if the targets were done correctly by the child.  Usually targets are correctly done if therapist appreciates the child. Also list targets that were not done correctly by the child. List which target was done how many time, and child responded how many time correctly and incorrectly. 
3. What are the area of improvements for a therapist?
4. Generate a summary in the end. 

- Mands, Spontaneous Verbalizations
- Tacting
- Intraverbals
- Listener Response / Receptive Language
-  Echoics
-  Imitation
-  Gross Motor Skills
-  Fine Motor Skills
-  Visual Perceptual and Matching to Sample
-  Disruptive Behaviors
-  Reinforcement and Praise 

This is a ABA therapy session text: 
Therapist is making the child ready for the session: OK, ready?
Therapist is giving instruction to the child. "Touch Banana"
Therapist is responding to child's response : No, try again.
Therapist is giving instruction to the child. "Touch Banana"
Therapist is responding to the child. "Very Nice"
Therapist is making the child ready for the session: "Ok Here we Go?"
Therapist is giving instruction to the child. "Touch Banana"
Therapist is responding to the child. "Good Job"
Therapist is giving instruction to the child. "Touch Banana"
Therapist is responding to the child. "That's right, banana."
Therapist is praising the child "Wooo!"
Therapist is giving instruction to the child. "Touch Banana"
Therapist is responding to the child. "Very good!"
Therapist is making the child ready for the session: "Three, two, one."?
Therapist is praising the child : "Agh, agh, agh, agh, agh, agh, agh."?
Therapist is giving instruction to the child. "OK, that's banana."
Therapist is responding to the child. "Excellent Job"
Therapist is responding to the child. "Good Job"'''
        inputs = tokenizer(prompt2, return_tensors="pt")
        logging.debug("Generating text.")
        generate_ids = model.generate(
            inputs.input_ids, max_length=len(prompt2))
        response = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        logging.debug("Generated response: %s", response)

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
