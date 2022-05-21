"""
Run experiment matrix, but time it. Thus, doesn't report to W&B
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from eval_stereoset import eval_stereoset, get_nsp_model
from generate_combinations import generate_combinations
from model import get_model
from os.path import exists
import sys
import json
import time
from dataset import AntiBiasTrainDataset, AntiBiasTestDataset, PADDING_ID
from transformers import GPT2Tokenizer

from training import train, eval
from training_args import TrainingArgs
from official_combinations import official_combinations, official_trials

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# trials = 2
trials = official_trials

"""
Specify every combination of various variables
"""

# variables = {
#     'num_epoches': [10],
#     'lr': [0.00001],
#     'batch_size': [50],
#     'optimizer': ['adam'],
#     'in_net': [False],
#     'in_net_init_identity': [False],
#     'out_net': [False],
#     'out_net_init_identity': [False],
#     'freeze_ln': [False],
#     'freeze_pos': [False],
#     'freeze_wte': [False],
#     'dup_lm_head': [False],
#     'dup_lm_head_bias': [False],
#     'freeze_ff': [False],
#     'freeze_attn': [False]
# }

# # These combinations don't make sense
# banned_combinations = [
# ]

# combinations = generate_combinations(
#     variables, list(variables.keys()), banned_combinations)

"""
Specify custom combinations
"""

# combinations = [
#     {
#         'num_epoches': 2,
#         'lr': 0.00055,
#         'batch_size': 50,
#         'optimizer': 'adam',
#         'in_net': True,
#         'in_net_init_identity': True,
#         'out_net': True,
#         'out_net_init_identity': True,
#         'freeze_ln': False,
#         'freeze_pos': False,
#         'freeze_wte': False,
#         'dup_lm_head': False,
#         'dup_lm_head_bias': False,
#         'freeze_ff': True,
#         'freeze_attn': True,
#     },
#     {
#         'num_epoches': 2,
#         'lr': 0.00065,
#         'batch_size': 50,
#         'optimizer': 'adam',
#         'in_net': True,
#         'in_net_init_identity': True,
#         'out_net': True,
#         'out_net_init_identity': True,
#         'freeze_ln': False,
#         'freeze_pos': False,
#         'freeze_wte': False,
#         'dup_lm_head': False,
#         'dup_lm_head_bias': False,
#         'freeze_ff': True,
#         'freeze_attn': True,
#     },
# ]

# for i in range(len(combinations)):
#     combinations[i]['model_save_path'] = f'matrix-results/unprejudiced-{i + 173}'

"""
For camera-ready experiments
"""

combinations = official_combinations

print(combinations)

for i, combination in enumerate(combinations):

    print(f'[{i + 1}/{len(combinations)} : {(i + 1)/len(combinations)*100:.3}%] Considering experiment with combination:')
    print(json.dumps(combination, indent=4))
    print('')

    trial_results = {}
    json_save_path = combination['model_save_path'] + '.json'

    if exists(json_save_path):
        print('Experiment already run! Skipping...')
        continue

    for trial in range(trials):
        # model_save_path = combination['model_save_path'] + \
        #     '-' + str(trial) + '.pth'
        model_save_path = combination['model_save_path'] + '.pth'
        try:
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
                'model_save_path': model_save_path,
                # Don't send to W&B since StereoSet eval takes up the most time
                'eval_stereoset': False,
                'write_stereoset': False,
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

            args = TrainingArgs(exp_args)

            # run = wandb.init(reinit=True, config=combination,
            #                  project='unprejudiced', entity='unprejudiced')

            start = time.time()
            train(TrainingArgs(exp_args), wandb_logging=False)
            stop = time.time()
            duration = stop - start

            print(f'Duration: {duration}')

            model = get_model(
                device=exp_args['device'], gpt2_name=exp_args['gpt2_name'],
                in_net=exp_args['in_net'], in_net_init_identity=exp_args['in_net_init_identity'],
                out_net=exp_args['out_net'], out_net_init_identity=exp_args['out_net_init_identity'],
                freeze_ln=exp_args['freeze_ln'], freeze_pos=exp_args['freeze_pos'], freeze_wte=exp_args['freeze_wte'],
                freeze_ff=exp_args['freeze_ff'], freeze_attn=exp_args['freeze_attn'],
                dup_lm_head=exp_args['dup_lm_head'], dup_lm_head_bias=exp_args['dup_lm_head_bias']
            )
            checkpoint = torch.load(model_save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to("cuda")

            nsp_head_model = get_nsp_model(
                model_save_path, './StereoSet/code/models/pretrained_models/GPT2Model_gpt2_0.0005.pth')

            # Evaluate validation, test loss
            validation_dataset = AntiBiasTestDataset(
                args.data_folder_path+args.validation_data_file, tokenizer, args.device)
            validation_dataloader = DataLoader(validation_dataset, 1)

            test_dataset = AntiBiasTestDataset(
                args.data_folder_path+args.test_data_file, tokenizer, args.device)
            test_dataloader = DataLoader(test_dataset, 1)

            validation_loss = eval(args, model, validation_dataloader)
            eval_loss = eval(args, model, test_dataloader)

            overall_results, intrasentence_bias, intersentence_bias = eval_stereoset(
                model, nsp_head_model, False, False)

            overall_results['metadata'] = {}

            overall_results['metadata']['duration'] = {}
            overall_results['metadata']['duration']['duration'] = duration

            overall_results['metadata']['losses'] = {}
            overall_results['metadata']['losses']['validation_loss'] = validation_loss
            overall_results['metadata']['losses']['eval_loss'] = eval_loss

            for sentence_type in overall_results:
                if sentence_type not in trial_results:
                    trial_results[sentence_type] = {}
                for category in overall_results[sentence_type]:
                    if category not in trial_results[sentence_type]:
                        trial_results[sentence_type][category] = {}
                    for score_type in overall_results[sentence_type][category]:
                        if score_type not in trial_results[sentence_type][category]:
                            trial_results[sentence_type][category][score_type] = [
                            ]
                        trial_results[sentence_type][category][score_type].append(
                            overall_results[sentence_type][category][score_type]
                        )

        except KeyboardInterrupt:
            sys.exit()
        except Exception as error:
            print(f'Run {i + 1} failed!')
            print(error)
        # finally:
        #     run.finish()

    # Calculate average/standard deviation of trials
    average = {}
    for sentence_type in trial_results:
        average[sentence_type] = {}
        for category in trial_results[sentence_type]:
            average[sentence_type][category] = {}
            for score_type in trial_results[sentence_type][category]:
                average[sentence_type][category][score_type] = np.average(
                    trial_results[sentence_type][category][score_type])

    std = {}
    for sentence_type in trial_results:
        std[sentence_type] = {}
        for category in trial_results[sentence_type]:
            std[sentence_type][category] = {}
            for score_type in trial_results[sentence_type][category]:
                std[sentence_type][category][score_type] = np.std(
                    trial_results[sentence_type][category][score_type])

    print('For combo:')
    print(json.dumps(combination, indent=4))
    print('Results:')
    print(json.dumps(trial_results, indent=4))
    print('Average:')
    print(json.dumps(average, indent=4))
    print('Std:')
    print(json.dumps(std, indent=4))

    with open(f'{json_save_path}', 'w') as outfile:
        json.dump({
            'combination': combination,
            'results': trial_results,
            'average': average,
            'std': std
        }, outfile, indent=4)
