
############# step 1 pretraining  #################

CUDA_VISIBLE_DEVICES=2 python pretrain_ptm_with_kg.py --num_epochs 20 —batch_size 8 —ratio_max_length 400

############# step 2 generate Embedding from my pretrained model #################

python generatePtmEmbeddings.py