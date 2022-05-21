from StereoSet.code.eval_generative_models import BiasEvaluator
from StereoSet.code.evaluation import ScoreEvaluator
# from StereoSet.code import evaluation
from model import get_model, tokenizer
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
import json
import os.path
import pathlib
import pyperclip
from collections import OrderedDict
from StereoSet.code.models import ModelNSP


class InterSentenceArg(object):
    def __init__(self, batch_size=1, input_file='StereoSet/data/dev.json', intersentence_load_path=None, intersentence_model='ModelNSP', intrasentence_load_path=None,
                 intrasentence_model='GPT2LM', max_seq_length=64, no_cuda=False, output_dir='predictions/', pretrained_class='gpt2', skip_intersentence=False,
                 skip_intrasentence=False, small=False, tokenizer='GPT2Tokenizer', unconditional_start_token='<|endoftext|>'):
        self.batch_size = batch_size
        self.input_file = input_file
        self.intersentence_load_path = intersentence_load_path
        self.intersentence_model = intersentence_model
        self.intrasentence_load_path = intrasentence_load_path
        self.intrasentence_model = intrasentence_model
        self.max_seq_length = max_seq_length
        self.no_cuda = no_cuda
        self.output_dir = output_dir
        self.pretrained_class = pretrained_class
        self.skip_intersentence = skip_intersentence
        self.skip_intrasentence = skip_intrasentence
        self.small = small
        self.tokenizer = tokenizer
        self.unconditional_start_token = unconditional_start_token


def eval_stereoset(lm_head_model, nsp_head_model, copy=False, eval_intersentence=True):

    dirname = os.path.dirname(__file__)
    input_file_path = os.path.join(dirname, 'StereoSet/data/dev.json')

    print('Dev JSON file path')
    print(input_file_path)

    evaluator = BiasEvaluator(
        trained_intra_model=lm_head_model, trained_inter_model=nsp_head_model, input_file=input_file_path, skip_intersentence=not eval_intersentence)

    intrasentence_bias = evaluator.evaluate_intrasentence()

    if eval_intersentence:
        intersentence_bias = evaluator.evaluate_nsp_intersentence(
            args=InterSentenceArg())
    else:
        base_eval_file_path = os.path.join(
            dirname, 'StereoSet/code/predictions/predictions_gpt2_ModelNSP_GPT2LM.json')
        with open(base_eval_file_path) as base_eval_file:
            template_file = json.load(base_eval_file)
            intersentence_bias = template_file['intersentence']

    bias = {}
    bias['intrasentence'] = intrasentence_bias
    bias['intersentence'] = intersentence_bias

    score_evaluator = ScoreEvaluator(input_file_path, bias)
    overall_results = score_evaluator.get_overall_results()

    if not eval_intersentence:
        overall_results.pop('intersentence')
        overall_results.pop('overall')
        intersentence_bias = None

    score_evaluator.pretty_print(overall_results)

    copy_scores = [
        [
            overall_results['intrasentence']['gender']['LM Score'],
            overall_results['intrasentence']['gender']['SS Score'],
            overall_results['intrasentence']['gender']['ICAT Score']
        ],
        [
            overall_results['intrasentence']['profession']['LM Score'],
            overall_results['intrasentence']['profession']['SS Score'],
            overall_results['intrasentence']['profession']['ICAT Score']
        ],
        [
            overall_results['intrasentence']['race']['LM Score'],
            overall_results['intrasentence']['race']['SS Score'],
            overall_results['intrasentence']['race']['ICAT Score']
        ],
        [
            overall_results['intrasentence']['religion']['LM Score'],
            overall_results['intrasentence']['religion']['SS Score'],
            overall_results['intrasentence']['religion']['ICAT Score']
        ],
        [
            overall_results['intrasentence']['overall']['LM Score'],
            overall_results['intrasentence']['overall']['SS Score'],
            overall_results['intrasentence']['overall']['ICAT Score']
        ],
        # [
        #     overall_results['intersentence']['gender']['LM Score'],
        #     overall_results['intersentence']['gender']['SS Score'],
        #     overall_results['intersentence']['gender']['ICAT Score']
        # ],
        # [
        #     overall_results['intersentence']['profession']['LM Score'],
        #     overall_results['intersentence']['profession']['SS Score'],
        #     overall_results['intersentence']['profession']['ICAT Score']
        # ],
        # [
        #     overall_results['intersentence']['race']['LM Score'],
        #     overall_results['intersentence']['race']['SS Score'],
        #     overall_results['intersentence']['race']['ICAT Score']
        # ],
        # [
        #     overall_results['intersentence']['religion']['LM Score'],
        #     overall_results['intersentence']['religion']['SS Score'],
        #     overall_results['intersentence']['religion']['ICAT Score']
        # ],
        # [
        #     overall_results['intersentence']['overall']['LM Score'],
        #     overall_results['intersentence']['overall']['SS Score'],
        #     overall_results['intersentence']['overall']['ICAT Score']
        # ],
    ]

    copy_string = ''
    for copy_category_scores in copy_scores:
        copy_string += '\t'.join([str(number)
                                 for number in copy_category_scores])
        copy_string += '\n'

    if copy:
        pyperclip.copy(copy_string)
        print('Scores copied for pasting in Google Sheets!')

    return overall_results, intrasentence_bias, intersentence_bias


def get_nsp_model(our_model_path, stereoset_nsp_model_path):
    return None
    # new_nsp_model_ordered_dict = OrderedDict()

    # # load our check point
    # our_checkpoint = torch.load(our_model_path)
    # our_model_state_dict = our_checkpoint['model_state_dict']
    # for key in our_model_state_dict:
    #     if "lm_head" not in key:
    #         new_nsp_model_ordered_dict[key.replace(
    #             "transformer", "core_model")] = our_model_state_dict[key]

    # # load stereo set nsp check point
    # stereoset_nsp_model_state_dict = torch.load(stereoset_nsp_model_path)
    # for key in stereoset_nsp_model_state_dict:
    #     if "nsp_head" in key:
    #         new_nsp_model_ordered_dict[key.replace(
    #             "module.", '')] = stereoset_nsp_model_state_dict[key]

    # model = ModelNSP("gpt2")
    # model.load_state_dict(new_nsp_model_ordered_dict)

    # return model


if __name__ == '__main__':
    # lm head model
    lm_head_model = get_model(device='cuda', gpt2_name='gpt2', in_net=False, in_net_init_identity=False, out_net=False, out_net_init_identity=False, freeze_ln=False, freeze_pos=True,
                              freeze_wte=True, freeze_ff=True, freeze_attn=True, dup_lm_head=False, dup_lm_head_bias=False)
    checkpoint = torch.load(
        'unprejudiced/unprejudiced-5-6.pth')
    lm_head_model.load_state_dict(checkpoint['model_state_dict'])
    model = lm_head_model.to("cuda")

    # nsp head head model
    nsp_head_model = get_nsp_model(
        'unprejudiced/unprejudiced-5-6.pth', './stereoset_pretrained_models/GPT2Model_gpt2_0.0005.pth')

    overall_results, intrasentence_bias, intersentence_bias = eval_stereoset(
        model, nsp_head_model, True)
