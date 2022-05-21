official_trials = 10

official_combinations = [
    # LN
    {
        'num_epoches': 6,
        'lr': 0.003,
        'batch_size': 50,
        'optimizer': 'adam',
        'in_net': False,
        'in_net_init_identity': False,
        'out_net': False,
        'out_net_init_identity': False,
        'freeze_ln': False,
        'freeze_pos': True,
        'freeze_wte': True,
        'dup_lm_head': False,
        'dup_lm_head_bias': False,
        'freeze_ff': True,
        'freeze_attn': True,
        'model_save_path': 'official-matrix-results/unprejudiced_ln'
    },
    # LN + WPE
    {
        'num_epoches': 6,
        'lr': 0.003,
        'batch_size': 50,
        'optimizer': 'adam',
        'in_net': False,
        'in_net_init_identity': False,
        'out_net': False,
        'out_net_init_identity': False,
        'freeze_ln': False,
        'freeze_pos': False,
        'freeze_wte': True,
        'dup_lm_head': False,
        'dup_lm_head_bias': False,
        'freeze_ff': True,
        'freeze_attn': True,
        'model_save_path': 'official-matrix-results/unprejudiced_ln_wpe'
    },
    # LN + WPE + WTE
    {
        'num_epoches': 2,
        'lr': 0.0005,
        'batch_size': 50,
        'optimizer': 'adam',
        'in_net': False,
        'in_net_init_identity': False,
        'out_net': False,
        'out_net_init_identity': False,
        'freeze_ln': False,
        'freeze_pos': False,
        'freeze_wte': False,
        'dup_lm_head': False,
        'dup_lm_head_bias': False,
        'freeze_ff': True,
        'freeze_attn': True,
        'model_save_path': 'official-matrix-results/unprejudiced_ln_wpe_wte'
    },
    # LN + WPE + WTE + Input/Output Layer
    {
        'num_epoches': 2,
        'lr': 0.0006,
        'batch_size': 50,
        'optimizer': 'adam',
        'in_net': True,
        'in_net_init_identity': True,
        'out_net': True,
        'out_net_init_identity': True,
        'freeze_ln': False,
        'freeze_pos': False,
        'freeze_wte': False,
        'dup_lm_head': False,
        'dup_lm_head_bias': False,
        'freeze_ff': True,
        'freeze_attn': True,
        'model_save_path': 'official-matrix-results/unprejudiced_ln_wpe_wte_io'
    },
    # Full Model
    {
        'num_epoches': 2,
        'lr': 0.0002,
        'batch_size': 50,
        'optimizer': 'adam',
        'in_net': False,
        'in_net_init_identity': False,
        'out_net': False,
        'out_net_init_identity': False,
        'freeze_ln': False,
        'freeze_pos': False,
        'freeze_wte': False,
        'dup_lm_head': False,
        'dup_lm_head_bias': False,
        'freeze_ff': False,
        'freeze_attn': False,
        'model_save_path': 'official-matrix-results/unprejudiced_full'
    },

]
