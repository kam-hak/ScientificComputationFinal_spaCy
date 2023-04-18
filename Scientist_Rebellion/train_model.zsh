#!/usr/bin/env zsh

python -m spacy init config xr_2_config.cfg --lang en --pipeline textcat --optimize efficiency
python -m spacy train xr_2_config.cfg --paths.train ./train.spacy --paths.validation ./validation.spacy --output ./output

# gpu version - not enough memory/time
#python -m spacy init config gpu_config.cfg --lang en --pipeline textcat --optimize efficiency
#python -m spacy train xr_gpu_config.cfg --paths.train ./train.spacy --paths.validation ./validation.spacy --gpu-id 0 --output ./output/gpu_model

# gpu version - doesn't work for bag of words - not enough memory/time
#python -m spacy init fill-config xr_gpu_config.cfg xr_gpu_config_pretrain.cfg --pretraining
#python -m spacy pretrain xr_gpu_config.cfg ./output_pretrain --paths.raw_text ./raw_text.spacy --gpu-id 0

# pretrain -- doesn't work for bag of words - cpu
#python -m spacy init fill-config xr_config.cfg xr_config_pretrain.cfg --pretraining
#python -m spacy pretrain xr_config.cfg ./output_pretrain --paths.raw_text ./raw_text.spacy 
