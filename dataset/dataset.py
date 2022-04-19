import torch
from torch.utils.data import Dataset

# Labels set to -100 are ignored (masked)
# https://huggingface.co/docs/transformers/master/model_doc/gpt2#transformers.GPT2DoubleHeadsModel.forward.labels
PADDING_ID = -100


class AntiBiasTrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, input_max_dim, device):
        super().__init__()
        self.data_path = data_path
        self.device = device
        self.tokenizer = tokenizer
        self.input_max_dim = input_max_dim

        # return a list of strs
        self.datafile = open(data_path, encoding='utf-8').readlines()

    def __len__(self):
        return len(self.datafile)

    def __getitem__(self, idx):
        """return:
            sentence_ids_padded: padded sentence ids in shape (batch_size, input_max_dim)
        """

        sentence_str = self.datafile[idx]
        sentence_ids = self.tokenizer(sentence_str)[
            'input_ids']  # a list of ids
        sentence_ids_padded = sentence_ids[:min(self.input_max_dim, len(
            sentence_ids))] + [PADDING_ID]*max(0, self.input_max_dim-len(sentence_ids))  # a list of padded ids
        sentence_ids_padded = torch.tensor(sentence_ids_padded).to(self.device)

        return sentence_ids_padded


"""
The difference from AntiBiasTrainDataset above is that AntiBiasTestDataset does not pad sentence
"""


class AntiBiasTestDataset(Dataset):
    def __init__(self, data_path, tokenizer, device):
        super().__init__()
        self.data_path = data_path
        self.device = device
        self.tokenizer = tokenizer

        # return a list of strs
        self.datafile = open(data_path, encoding='utf-8').readlines()

    def __len__(self):
        return len(self.datafile)

    def __getitem__(self, idx):
        """return:
            sentence_ids_padded: padded sentence ids in shape (batch_size, input_max_dim)
        """

        sentence_str = self.datafile[idx]
        sentence_ids = torch.tensor(self.tokenizer(sentence_str)[
                                    'input_ids']).to(self.device)  # a list of ids
        return sentence_ids
