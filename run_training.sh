python training.py \
        --device 'cpu' \
        --num_epoches 1 \
        --batch_size 1 \
        --lr 0.001 \
        --gpt_model_name 'gpt2' \
        --input_max_dim 50 \
        --model_save_path 'trained_models/model.pth' \
        --load_pretrained_model \
        --model_load_path 'trained_models/model.pth' \