#!/bin/bash

declare -a list_of_finetune_lr=(1e-5 3e-5 5e-5 7e-5)

for ((which_finetune_lr=0;which_finetune_lr<${#list_of_finetune_lr[@]};++which_finetune_lr)); do
    python finetuning_baseline.py --override False \
        --experiment marc_ckpts \
        --ptl bert \
        --model bert-base-multilingual-cased \
        --dataset_name marc \
        --trn_languages english \
        --eval_languages german,english,spanish,french,chinese,japanese \
        --finetune_epochs 10 \
        --finetune_batch_size 32 \
        --eval_every_batch 50 \
        --finetune_lr ${list_of_finetune_lr[which_finetune_lr]} \
        --manual_seed 42 \
        --train_fast False \
        --world 0,1,2,3
done
