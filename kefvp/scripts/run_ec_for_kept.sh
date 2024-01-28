#!/bin/bash

b=(3 7 15 30)
for dur in ${b[*]}
do
     a=(0.95)
     for thres in ${a[*]}
     do
     # EC
     CUDA_VISIBLE_DEVICES=2 python final_series_infer.py --weight_decay 5e-2 --lr 0.0002 --epochs 200 --pid ${thres} --time_model CondAutoformer --duration ${dur} --run_mode reg --d_layers 2\
          --dist_threshold ${thres} --mu 0.7 --dataset ec --audio_indim 768 --text_indim 1024 --text_embedding emnlp_202308_bert_large_unfreeze_6layers/ec_embed_bert_large_uncased_kept_epoch_6 \
          --raw_data_path /home/niuhao/project/html_www2020/raw_data/ReleasedDataset_mp3/ --pkl_save_path save_pkls/emnlp_202308_bert_large_unfreeze_6layers_ep6 --log_save_path emnlp_202308_bert_large_unfreeze_6layers_ep6       #不加音频
     done
done