from transformers import AutoTokenizer, AutoModelForCausalLM

access_token = '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'

tokenizer = AutoTokenizer.from_pretrained(
    "klyang/MentaLLaMA-chat-7B", legacy=False)
model = AutoModelForCausalLM.from_pretrained(
    "klyang/MentaLLaMA-chat-7B", token=access_token)

prompt = '''Analyze this instruction in an ABA session and tell me what is it Token economy
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
Therapist is giving instruction to the child. "Touch Banana"
"'''

# prompt3 = '''Therapist instructions such as DO this and Follow me form in which domain or verbal operant of ABA'''
prompt3 = '''Can you define verbal operands of ABA Therapy with some examples?'''
inputs = tokenizer(prompt3, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=len(prompt3))
aa = tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Response", aa)
