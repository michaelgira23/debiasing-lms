
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Prompt to feed the model
prompt = 'The doctor was successful because'

# How many responses to include
generate_n_responses = 3

# What device to run on
device = 'cuda'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

print(f'{"=" * 20}[Prompt]{"=" * 20}')
print(f'"{prompt}"')

encoded_prompt = tokenizer.encode(
    prompt, add_special_tokens=False, return_tensors="pt").to(device)

for i in range(generate_n_responses):
    print(f'{"=" * 20}[Response {i + 1}]{"=" * 20}')
    output_sequences = model.generate(
        input_ids=encoded_prompt, do_sample=True)

    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    text = tokenizer.decode(
        output_sequences[0], clean_up_tokenization_spaces=True)

    print(text)
