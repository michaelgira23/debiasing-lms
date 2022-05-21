python  /eval_generative_models.py \
        --pretrained-class gpt2 \
        --intrasentence-model GPT2LM \
        --intersentence-model ModelNSP \
        --tokenizer GPT2Tokenizer \
        --max-seq-length 128 \
        --intersentence-load-path models/pretrained_models/GPT2Model_gpt2_0.0005.pth \
        --batch-size 1 \
        --input-file ../data/dev.json \
        --output-dir predictions/
