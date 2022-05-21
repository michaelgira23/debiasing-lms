
from model import get_model, tokenizer
import torch
import json

from training_args import TrainingArgs

# Prompt to feed the model
prompt = 'The doctor was successful because'

# How many responses to include
generate_n_responses = 3

# Do not include file extension
model_save_path = 'official-matrix-results/unprejudiced_ln'

# Probably don't need to touch this
model_save_path_json = model_save_path + '.json'
model_save_path_pth = model_save_path + '.pth'

# What device to run on
device = 'cuda'

print(f'{"=" * 20}[Prompt]{"=" * 20}')
print(f'Prompt: "{prompt}"')
print(f'Model: {model_save_path}')

with open(model_save_path_json, encoding='utf-8', mode='r') as f:
    data = json.load(f)
    combination = data['combination']

    exp_args = TrainingArgs({
        'device': device,
        'num_epoches': combination['num_epoches'],
        'batch_size': combination['batch_size'],
        'lr': combination['lr'],
        'optimizer': combination['optimizer'],
        'input_max_dim': 50,
        'data_folder_path': 'unprejudiced_dataset/',
        'train_data_file': 'train.txt',
        'validation_data_file': 'validation.txt',
        'test_data_file': 'test.txt',
        'load_pretrained_model': False,
        'model_load_path': model_save_path_pth,
        'model_save_path': model_save_path_pth,
        'eval_stereoset': True,
        'write_stereoset': True,
        'gpt2_name': 'gpt2',
        'in_net': combination['in_net'],
        'in_net_init_identity': combination['in_net_init_identity'],
        'out_net': combination['out_net'],
        'out_net_init_identity': combination['out_net_init_identity'],
        'freeze_ln': combination['freeze_ln'],
        'freeze_pos': combination['freeze_pos'],
        'freeze_wte': combination['freeze_wte'],
        'freeze_ff': combination['freeze_ff'],
        'freeze_attn': combination['freeze_attn'],
        'dup_lm_head': combination['dup_lm_head'],
        'dup_lm_head_bias': combination['dup_lm_head_bias']
    })

    model = get_model(gpt2_name=exp_args.gpt2_name, in_net=exp_args.in_net, in_net_init_identity=exp_args.in_net_init_identity, out_net=exp_args.out_net,
                      out_net_init_identity=exp_args.out_net_init_identity, freeze_ln=exp_args.freeze_ln, freeze_pos=exp_args.freeze_pos,
                      freeze_wte=exp_args.freeze_wte, freeze_ff=exp_args.freeze_ff, freeze_attn=exp_args.freeze_attn, dup_lm_head=exp_args.dup_lm_head, dup_lm_head_bias=exp_args.dup_lm_head_bias)
    model = model.to(exp_args.device)

    checkpoint = torch.load(model_save_path_pth)
    model.load_state_dict(checkpoint['model_state_dict'])

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
