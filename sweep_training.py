"""
Run training.py for Weights and Biases Sweep hyperparameter tuning
"""
import wandb

from training import train
from training_args import TrainingArgs

combination = {
    'num_epoches': 8,
    'lr': 10 ** -3,
    'batch_size': 25,
    'optimizer': 'adam',
    'in_net': True,
    'in_net_init_identity': False,
    'out_net': True,
    'out_net_init_identity': False,
    'freeze_ln': True,
    'freeze_pos': True,
    'freeze_wte': True,
    'dup_lm_head': True,
    'dup_lm_head_bias': True
}

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

run = wandb.init(config=exp_args)

config = wandb.config

train(TrainingArgs(config), wandb_logging=True)
