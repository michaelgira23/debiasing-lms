import math
from numpy import NaN
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from model import get_model
from dataset import AntiBiasTrainDataset, AntiBiasTestDataset, PADDING_ID

from training import eval
from training_args import TrainingArgs


path = './matrix-results/'
save_results = './matrix.csv'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

matrix = pd.read_csv(save_results)

if 'validation_loss' not in matrix.columns:
    matrix.insert(0, 'validation_loss', None)

for index, row in matrix.iterrows():

    if not math.isnan(row['validation_loss']):
        continue

    model_save_path = row['combination.model_save_path'] + '.pth'

    exp_args = TrainingArgs({
        'device': 'cuda',
        'num_epoches': row['combination.num_epoches'],
        'batch_size': row['combination.batch_size'],
        'lr': row['combination.lr'],
        'optimizer': row['combination.optimizer'],
        'input_max_dim': 50,
        'data_folder_path': 'unprejudiced_dataset/',
        'train_data_file': 'train.txt',
        'validation_data_file': 'validation.txt',
        'test_data_file': 'test.txt',
        'load_pretrained_model': False,
        'model_load_path': model_save_path,
        'model_save_path': model_save_path,
        'eval_stereoset': True,
        'write_stereoset': True,
        'gpt2_name': 'gpt2',
        'in_net': row['combination.in_net'],
        'in_net_init_identity': row['combination.in_net_init_identity'],
        'out_net': row['combination.out_net'],
        'out_net_init_identity': row['combination.out_net_init_identity'],
        'freeze_ln': row['combination.freeze_ln'],
        'freeze_pos': row['combination.freeze_pos'],
        'freeze_wte': row['combination.freeze_wte'],
        'freeze_ff': True if 'combination.freeze_ff' not in row else row['combination.freeze_ff'],
        'freeze_attn': True if 'combination.freeze_attn' not in row else row['combination.freeze_attn'],
        'dup_lm_head': row['combination.dup_lm_head'],
        'dup_lm_head_bias': row['combination.dup_lm_head_bias']
    })

    model = get_model(gpt2_name=exp_args.gpt2_name, in_net=exp_args.in_net, in_net_init_identity=exp_args.in_net_init_identity, out_net=exp_args.out_net,
                      out_net_init_identity=exp_args.out_net_init_identity, freeze_ln=exp_args.freeze_ln, freeze_pos=exp_args.freeze_pos,
                      freeze_wte=exp_args.freeze_wte, freeze_ff=exp_args.freeze_ff, freeze_attn=exp_args.freeze_attn, dup_lm_head=exp_args.dup_lm_head, dup_lm_head_bias=exp_args.dup_lm_head_bias)
    model = model.to(exp_args.device)

    checkpoint = torch.load(exp_args.model_load_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    validation_dataset = AntiBiasTestDataset(
        exp_args.data_folder_path+exp_args.validation_data_file, tokenizer, exp_args.device)
    validation_dataloader = DataLoader(validation_dataset, 1)

    validation_loss = eval(None, model, validation_dataloader)

    matrix.at[index, 'validation_loss'] = validation_loss
    matrix.to_csv(save_results, index=False)

    print(f'[{index + 1}/{matrix.shape[0]} : {(index + 1)/matrix.shape[0]*100:.3}%] Validation loss: {validation_loss}')

print(matrix)
