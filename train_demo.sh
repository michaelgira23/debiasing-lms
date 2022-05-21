python training.py \
    --device cuda \
    --num_epoches 4 \
    --batch_size 10 \
    --lr 0.003898 \
    --input_max_dim 50 \
    --model_save_path 'unprejudiced/unprejudiced-6.pth' \
    --freeze_pos \
    --freeze_wte \
    --freeze_ff \
    --freeze_attn

# Original (with broken '!' padding)
# python training.py \
#     --device cuda \
#     --num_epoches 8 \
#     --batch_size 100 \
#     --lr 0.001 \
#     --input_max_dim 50 \
#     --model_save_path 'unprejudiced/unprejudiced.pth' \
#     --in_net \
#     # --in_net_init_identity \
#     # --out_net \
#     # --out_net_init_identity \
#     # --freeze_ln \
#     # --freeze_pos \
#     --freeze_wte \
#     --freeze_ff \
#     --freeze_attn
#     # --dup_lm_head \
#     # --dup_lm_head_bias
