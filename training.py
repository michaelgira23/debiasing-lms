import torch
from torch.utils.data import Dataset, DataLoader
from dataset import AntiBiasTrainDataset, AntiBiasTestDataset, PADDING_ID
import argparse
import csv
from datetime import datetime
import os
import sys

from transformers import GPT2Tokenizer
# from model import model as untrained_model
from model import get_model
import numpy as np
from eval_stereoset import eval_stereoset, get_nsp_model

try:
    import wandb
except:
    pass


def eval(exp_args, model, dataloader):
    model.eval()
    steps = 0
    avg_loss = .0
    for i_batch, batched_ids in enumerate(dataloader):
        output = model(batched_ids, labels=batched_ids)
        loss = output.loss
        avg_loss += loss.detach().cpu().item()
        steps += 1
    avg_loss /= steps
    return avg_loss


def train_epoch(exp_args, model, dataloader, optimizer):
    model.train()
    steps = 0
    avg_loss = .0
    for i_batch, batched_ids in enumerate(dataloader):

        # Calculate attention mask
        mask = (batched_ids != PADDING_ID).int()
        batched_ids_positive = torch.where(
            batched_ids == PADDING_ID, 0, batched_ids)

        optimizer.zero_grad()

        output = model(batched_ids_positive,
                       labels=batched_ids, attention_mask=mask)
        # lm_logits = output.lm_logits
        loss = output.loss
        avg_loss += loss.detach().cpu().item()
        loss.backward()
        optimizer.step()
        steps += 1
    avg_loss /= steps
    return avg_loss


def train(exp_args, wandb_logging=False):
    # initialize dataset
    print(exp_args)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    train_dataset = AntiBiasTrainDataset(
        exp_args.data_folder_path+exp_args.train_data_file, tokenizer, exp_args.input_max_dim, exp_args.device)
    train_dataloader = DataLoader(train_dataset, exp_args.batch_size)

    validation_dataset = AntiBiasTestDataset(
        exp_args.data_folder_path+exp_args.validation_data_file, tokenizer, exp_args.device)
    validation_dataloader = DataLoader(validation_dataset, 1)

    test_dataset = AntiBiasTestDataset(
        exp_args.data_folder_path+exp_args.test_data_file, tokenizer, exp_args.device)
    test_dataloader = DataLoader(test_dataset, 1)

    # initialize model and load pretrained model as need
    model = get_model(gpt2_name=exp_args.gpt2_name, in_net=exp_args.in_net, in_net_init_identity=exp_args.in_net_init_identity, out_net=exp_args.out_net,
                      out_net_init_identity=exp_args.out_net_init_identity, freeze_ln=exp_args.freeze_ln, freeze_pos=exp_args.freeze_pos,
                      freeze_wte=exp_args.freeze_wte, freeze_ff=exp_args.freeze_ff, freeze_attn=exp_args.freeze_attn, dup_lm_head=exp_args.dup_lm_head, dup_lm_head_bias=exp_args.dup_lm_head_bias)

    if wandb_logging:
        wandb.watch(model, log_freq=100)

    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name, p.requires_grad)
    # for p in model.lm_head.parameters():
    #     print('lm_head', p.requires_grad)

    model = model.to(exp_args.device)
    if exp_args.load_pretrained_model:
        try:
            checkpoint = torch.load(exp_args.model_load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("successfully loaded model from", exp_args.model_load_path)
        except:
            print("failed to load model from", exp_args.model_load_path)

    if exp_args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=exp_args.lr)

    # Evaluate the intial model
    if wandb_logging:
        validation_loss = eval(exp_args, model, validation_dataloader)
        eval_loss = eval(exp_args, model, test_dataloader)
        print("Training Epoch: 0")
        print("Training Loss: NaN")
        print("Validation Loss:", validation_loss)
        print("Testing Loss:", eval_loss)

    loss_list = []
    last_loss = 0

    best_train_loss = float('inf')
    best_validation_loss = float('inf')
    best_eval_loss = float('inf')

    best_icat = float('-inf')
    best_icat_index = float('-inf')
    best_train_loss_for_icat = float('inf')
    best_eval_loss_for_icat = float('inf')

    best_lm = float('-inf')
    best_lm_index = float('-inf')
    best_train_loss_for_lm = float('inf')
    best_eval_loss_for_lm = float('inf')

    best_ss = float('-inf')
    best_ss_index = float('-inf')
    best_train_loss_for_ss = float('inf')
    best_eval_loss_for_ss = float('inf')

    if exp_args.eval_stereoset and exp_args.write_stereoset:
        now = datetime.now()
        dirname = os.path.dirname(__file__)
        stereoset_results_path = os.path.join(
            dirname, f'results/{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}.csv')
        wrote_header = False

    for epoch in range(1, exp_args.num_epoches+1):
        train_loss = train_epoch(exp_args, model, train_dataloader, optimizer)

        # This probably isn't the right parameter to check but it correlationally works
        if wandb_logging:
            validation_loss = eval(exp_args, model, validation_dataloader)
            eval_loss = eval(exp_args, model, test_dataloader)

            # print out training info
            print("Training Epoch:", epoch)
            print("Training Loss:", train_loss)
            print("Validation Loss:", validation_loss)
            print("Testing Loss:", eval_loss)

            epoch_results = {
                'epoch': epoch,
                'train_loss': train_loss,
                'validation_loss': validation_loss,
                'eval_loss': eval_loss
            }

            loss_list.append([train_loss, eval_loss])

        # save trained model per epoch
        state_dict = dict(epoch=epoch, loss=train_loss,
                          model_state_dict=model.state_dict())
        try:
            torch.save(state_dict, exp_args.model_save_path)
            print("Successfully saved model after", epoch,
                  "epoches at", exp_args.model_save_path)
        except:
            torch.save(state_dict, './trained_model.pth')
            print("Successfully saved model after",
                  epoch, "epoches at ./trained_model.pth")

        if exp_args.eval_stereoset:
            print('Eval StereoSet')
            nsp_head_model = get_nsp_model(
                exp_args.model_save_path, './StereoSet/code/models/pretrained_models/GPT2Model_gpt2_0.0005.pth')

            overall_results, intrasentence_bias, intersentence_bias = eval_stereoset(
                model, nsp_head_model, False, False)

            if exp_args.write_stereoset:

                for sentence_type in overall_results:
                    for category in overall_results[sentence_type]:
                        # For the 'overall' score at the end
                        if isinstance(overall_results[sentence_type][category], float):
                            epoch_results[f'{sentence_type}_{category}'] = overall_results[sentence_type][category]
                        else:
                            for score_type in overall_results[sentence_type][category]:
                                epoch_results[f'{sentence_type}_{category}_{score_type}'] = overall_results[sentence_type][category][score_type]

                index = epoch - 1

                sample_icat = epoch_results['intrasentence_overall_ICAT Score']
                sample_lm = epoch_results['intrasentence_overall_LM Score']
                sample_ss = epoch_results['intrasentence_overall_SS Score']

                best_train_loss = min(best_train_loss, train_loss)
                best_validation_loss = min(
                    best_validation_loss, validation_loss)
                best_eval_loss = min(best_eval_loss, eval_loss)

                if sample_icat > best_icat:
                    best_icat = sample_icat
                    best_icat_index = index
                    best_train_loss_for_icat = train_loss
                    best_validation_loss_for_icat = validation_loss
                    best_eval_loss_for_icat = eval_loss

                if sample_lm > best_lm:
                    best_lm = sample_lm
                    best_lm_index = index
                    best_train_loss_for_lm = train_loss
                    best_validation_loss_for_lm = validation_loss
                    best_eval_loss_for_lm = eval_loss

                if abs(sample_ss - 50) < abs(best_ss - 50):
                    best_ss = sample_ss
                    best_ss_index = index
                    best_train_loss_for_ss = train_loss
                    best_validation_loss_for_ss = validation_loss
                    best_eval_loss_for_ss = eval_loss

                with open(stereoset_results_path, 'a', newline='') as f:
                    writer = csv.DictWriter(
                        f, fieldnames=epoch_results.keys())

                    if not wrote_header:
                        writer.writeheader()
                        wrote_header = True

                    writer.writerow(epoch_results)

                if wandb_logging:
                    wandb.run.summary['best_lm'] = best_lm
                    wandb.run.summary['best_ss'] = best_ss
                    wandb.run.summary['best_icat'] = best_icat

                    wandb.run.summary['min_train_loss'] = best_train_loss
                    wandb.run.summary['min_validation_loss'] = best_validation_loss
                    wandb.run.summary['min_eval_loss'] = best_eval_loss

                    wandb.run.summary['best_icat_index'] = best_icat_index
                    wandb.run.summary['best_train_loss_for_icat'] = best_train_loss_for_icat
                    wandb.run.summary['best_validation_loss_for_icat'] = best_validation_loss_for_icat
                    wandb.run.summary['best_eval_loss_for_icat'] = best_eval_loss_for_icat

                    wandb.run.summary['best_lm_index'] = best_lm_index
                    wandb.run.summary['best_train_loss_for_lm'] = best_train_loss_for_lm
                    wandb.run.summary['best_validation_loss_for_lm'] = best_validation_loss_for_lm
                    wandb.run.summary['best_eval_loss_for_lm'] = best_eval_loss_for_lm

                    wandb.run.summary['best_ss_index'] = best_ss_index
                    wandb.run.summary['best_train_loss_for_ss'] = best_train_loss_for_ss
                    wandb.run.summary['best_validation_loss_for_ss'] = best_validation_loss_for_ss
                    wandb.run.summary['best_eval_loss_for_ss'] = best_eval_loss_for_ss

        if wandb_logging:
            wandb.log(epoch_results)

        if epoch > 1:
            # if last_loss - train_loss <= 0.001 or epoch == exp_args.num_epoches:
            if epoch == exp_args.num_epoches:
                if wandb_logging:
                    try:
                        np.savetxt(exp_args.model_save_path+".csv",
                                   np.array(loss_list), delimiter=",")
                    except:
                        pass
                print("Stop training after " + str(epoch)+" epoches.")
                break
            else:
                last_loss = train_loss
        else:
            last_loss = train_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_epoches', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--optimizer', type=str,
                        default='adam', choices=['adam'])

    parser.add_argument('--input_max_dim', type=int, default=50)

    parser.add_argument('--data_folder_path', type=str,
                        default='unprejudiced_dataset/')
    parser.add_argument('--train_data_file', type=str, default='train.txt')
    parser.add_argument('--validation_data_file',
                        type=str, default='validation.txt')
    parser.add_argument('--test_data_file', type=str, default='test.txt')

    parser.add_argument('--load_pretrained_model', action='store_true')
    parser.add_argument('--model_load_path', type=str,
                        default='trained_models/model.pth')
    parser.add_argument('--model_save_path', type=str,
                        default='trained_models/model.pth')

    parser.add_argument('--eval_stereoset', action='store_true')
    parser.add_argument('--write_stereoset', action='store_true')

    parser.add_argument('--gpt2_name', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--in_net', action='store_true')
    parser.add_argument('--in_net_init_identity', action='store_true')
    parser.add_argument('--out_net', action='store_true')
    parser.add_argument('--out_net_init_identity', action='store_true')
    parser.add_argument('--freeze_ln', action='store_true')
    parser.add_argument('--freeze_pos', action='store_true')
    parser.add_argument('--freeze_wte', action='store_true')
    parser.add_argument('--freeze_ff', action='store_true')
    parser.add_argument('--freeze_attn', action='store_true')

    parser.add_argument('--dup_lm_head', action='store_true')
    parser.add_argument('--dup_lm_head_bias', action='store_true')

    exp_args = parser.parse_args(sys.argv[1:])
    train(exp_args)
