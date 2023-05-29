import json
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

# Load the pre-trained GPT-3.5 model and tokenizer
model_name = 'EleutherAI/gpt-neo-1.3B'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Example JSON data
json_examples = {
    "create instances": {"ec2_count": 1},
    "create instances with name {name}": {"ec2_count": 1, "instance_names": "{name}"},
}

def generate_response(input_text):
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate a response using the pre-trained model
    output = model.generate(input_ids, max_length=100)

    # Decode the generated output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def process_input(input_text):
    for example, response in json_examples.items():
        # Replace placeholders with actual values
        if '{name}' in example:
            name = input_text.split('with name ')[-1]
            replaced_example = example.replace('{name}', name)
            response = json.dumps(response).replace('{name}', name)  # Replace {name} in the response as well
        else:
            replaced_example = example

        # Check if the input matches the example
        if replaced_example == input_text:
            return response

    # If no match is found, generate a response using the language model
    return generate_response(input_text)


# Example usage
result = process_input("create instances with name farhan")
print(result)
