a=(0.95 0.85 0.75 0.65 0.55 0.45 0.35)
for thres in ${a[*]}
do
# Maec-16
CUDA_VISIBLE_DEVICES=1 python final_series_infer.py --weight_decay 5e-2 --lr 0.0002 --epochs 200 --pid ${thres} --time_model CondAutoformer --duration 3 --run_mode reg --d_layers 2\
     --dist_threshold ${thres} --mu 0.7 --dataset 16 --audio_indim 29 --text_embedding raw_bert_base_uncased --raw_data_path html_www2020/MAECdata/MAEC_Dataset/ --root_dir /nfsfile/niuhao/project/       #不加音频
done