"""
Run a bunch of different experiments
"""

import json
import wandb

from training import train
from training_args import TrainingArgs

api = wandb.Api()

# Key = parameter to tweak
# Value = list of values to take on. Each combination will have
variables = {
    'num_epoches': [8],
    'lr': list(map(lambda exp: 10 ** exp, range(-6, -2))),
    'optimizer': ['adam'],
    'in_net': [True, False],
    'in_net_init_identity': [True, False],
    'out_net': [True, False],
    'out_net_init_identity': [True, False],
    'freeze_ln': [True, False],
    'freeze_pos': [True, False],
    'freeze_wte': [True, False],
    'dup_lm_head': [True, False],
    'dup_lm_head_bias': [True, False]
}

# These combinations don't make sense
banned_combinations = [
    {
        'in_net': False,
        'in_net_init_identity': True
    },
    {
        'out_net': False,
        'out_net_init_identity': True
    },
    {
        'dup_lm_head': False,
        'dup_lm_head_bias': True
    },
    # Gives error for some reason about gradients
    {
        'dup_lm_head_bias': False,
        'dup_lm_head': False,
        'freeze_wte': True,
        'freeze_pos': True,
        'freeze_ln': True,
        'out_net_init_identity': False,
        'out_net': False,
        'in_net_init_identity': False,
        'in_net': False
    }
]


def filter_combinations(criteria):
    def filter_combination_func(check):
        available_keys = set(check.keys())

        for banned_combination in criteria:
            if set(banned_combination.keys()) <= available_keys:
                is_different = False
                for combination_key in banned_combination:
                    if banned_combination[combination_key] != check[combination_key]:
                        is_different = True
                        break
                # Criteria exists in the combination in question
                if not is_different:
                    return False
        return True

    return filter_combination_func


def generate_combinations(reference_variables, keys, disallow_combinations):
    if not len(keys):
        return [{}]

    tweak_key = keys.pop(0)
    suffixes = generate_combinations(
        reference_variables, keys, disallow_combinations)

    total_combinations = []
    for suffix in suffixes:
        for value in reference_variables[tweak_key]:
            new_suffix = suffix.copy()
            new_suffix[tweak_key] = value
            total_combinations.append(new_suffix)

    # Filter out blacklist of combinations
    # Without filter: 2560 combinations
    # With filter: 1080 combinations (~58% reduction)
    return list(filter(filter_combinations(disallow_combinations), total_combinations))


combinations = generate_combinations(
    variables, list(variables.keys()), banned_combinations)

print(f'There are {len(combinations)} total experiment combinations')
# print(json.dumps(combinations[0:3], indent=4))


for i in range(600, len(combinations)):
    combination = combinations[i]

    print(f'[{i + 1}/{len(combinations)} : {(i + 1)/len(combinations)*100:.3}%] Considering experiment with combination:')
    print(json.dumps(combination, indent=4))
    print('')

    # Check if experiment combination has already run
    runs = api.runs("unprejudiced/unprejudiced")

    should_skip = False

    for run in runs:
        is_different = False
        for variable in combination:
            if run.config[variable] != combination[variable]:
                is_different = True
                break
        if not is_different:
            if 'epoch' in run.summary and run.summary['epoch'] == combination['num_epoches']:
                should_skip = True
                break

    if should_skip:
        print('Experiment already run! Skipping...')
        continue

    try:
        exp_args = {
            'device': 'cuda',
            'num_epoches': combination['num_epoches'],
            'batch_size': 100,
            'lr': combination['lr'],
            'optimizer': combination['optimizer'],
            'input_max_dim': 50,
            'data_folder_path': 'unprejudiced_dataset/',
            'train_data_file': 'train.txt',
            'validation_data_file': 'validation.txt',
            'test_data_file': 'test.txt',
            'load_pretrained_model': False,
            'model_load_path': '',
            'model_save_path': 'trained_models/experiments/model.pth',
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
            'freeze_ff': True,
            'freeze_attn': True,
            'dup_lm_head': combination['dup_lm_head'],
            'dup_lm_head_bias': combination['dup_lm_head_bias']
        }

        run = wandb.init(reinit=True, config=combination,
                         project='unprejudiced', entity='unprejudiced')

        train(TrainingArgs(exp_args), wandb_logging=True)

    except:
        print(f'Run {i + 1} failed!')
    finally:
        run.finish()
