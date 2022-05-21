from dataclasses import dataclass

"""
Boilerplate to convert a dict to a dataclass
(required for the training.py train() function)
"""


@dataclass
class TrainingArgs:

    device: str
    num_epoches: int
    batch_size: int
    lr: float
    optimizer: str

    input_max_dim: int

    data_folder_path: str
    train_data_file: str
    validation_data_file: str
    test_data_file: str

    load_pretrained_model: bool
    model_load_path: str
    model_save_path: str

    eval_stereoset: bool
    write_stereoset: bool

    gpt2_name: str
    in_net: bool
    in_net_init_identity: bool
    out_net: bool
    out_net_init_identity: bool
    freeze_ln: bool
    freeze_pos: bool
    freeze_wte: bool
    freeze_ff: bool
    freeze_attn: bool

    dup_lm_head: bool
    dup_lm_head_bias: bool

    def __init__(self, args) -> None:
        self.device = args['device']
        self.num_epoches = args['num_epoches']
        self.batch_size = args['batch_size']
        self.lr = args['lr']
        self.optimizer = args['optimizer']

        self.input_max_dim = args['input_max_dim']

        self.data_folder_path = args['data_folder_path']
        self.train_data_file = args['train_data_file']
        self.validation_data_file = args['validation_data_file']
        self.test_data_file = args['test_data_file']

        self.load_pretrained_model = args['load_pretrained_model']
        self.model_load_path = args['model_load_path']
        self.model_save_path = args['model_save_path']

        self.eval_stereoset = args['eval_stereoset']
        self.write_stereoset = args['write_stereoset']

        self.gpt2_name = args['gpt2_name']
        self.in_net = args['in_net']
        self.in_net_init_identity = args['in_net_init_identity']
        self.out_net = args['out_net']
        self.out_net_init_identity = args['out_net_init_identity']
        self.freeze_ln = args['freeze_ln']
        self.freeze_pos = args['freeze_pos']
        self.freeze_wte = args['freeze_wte']
        self.freeze_ff = args['freeze_ff']
        self.freeze_attn = args['freeze_attn']

        self.dup_lm_head = args['dup_lm_head']
        self.dup_lm_head_bias = args['dup_lm_head_bias']
