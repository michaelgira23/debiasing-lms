import json
from official_combinations import official_combinations, official_trials
from training_args import TrainingArgs
from model import get_model

combinations = official_combinations

for i, combination in enumerate(combinations):
    print(f'[{i + 1}/{len(combinations)} : {(i + 1)/len(combinations)*100:.3}%] Considering model with combination:')
    print(json.dumps(combination, indent=4))
    print('')

    exp_args = {
        'device': 'cuda',
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
        'model_load_path': '',
        'model_save_path': '',
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
    }

    model = get_model(
        device=exp_args['device'], gpt2_name=exp_args['gpt2_name'],
        in_net=exp_args['in_net'], in_net_init_identity=exp_args['in_net_init_identity'],
        out_net=exp_args['out_net'], out_net_init_identity=exp_args['out_net_init_identity'],
        freeze_ln=exp_args['freeze_ln'], freeze_pos=exp_args['freeze_pos'], freeze_wte=exp_args['freeze_wte'],
        freeze_ff=exp_args['freeze_ff'], freeze_attn=exp_args['freeze_attn'],
        dup_lm_head=exp_args['dup_lm_head'], dup_lm_head_bias=exp_args['dup_lm_head_bias']
    )

    total_parameters = 0
    frozen_parameters = 0
    unfrozen_parameters = 0

    for name, p in model.named_parameters():
        size = p.size()
        param_count = 1
        for dimension in size:
            param_count *= dimension

        total_parameters += param_count

        if p.requires_grad:
            # print(f'{name} - unfrozen!')
            unfrozen_parameters += param_count
        else:
            # print(f'{name} - frozen!')
            frozen_parameters += param_count

    print(f'Total params: {total_parameters}')
    print(
        f'Frozen params: {frozen_parameters} ({frozen_parameters / total_parameters * 100:.2f}%)')
    print(
        f'Unfrozen params: {unfrozen_parameters} ({unfrozen_parameters / total_parameters * 100:.2f}%)')
