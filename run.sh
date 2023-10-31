CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv \
                                      --dataset tcga_blca \
                                      --data_root_dir /master/zhou_feng_tao/data/TCGA/BLCA/x20 \
                                      --modal coattn \
                                      --model cmta \
                                      --num_epoch 30 \
                                      --batch_size 1 \
                                      --loss nll_surv_l1 \
                                      --lr 0.001 \
                                      --optimizer SGD \
                                      --scheduler None \
                                      --alpha 1.0


CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv \
                                      --dataset tcga_brca \
                                      --data_root_dir /master/zhou_feng_tao/data/TCGA/BRCA/x20 \
                                      --modal coattn \
                                      --model cmta \
                                      --num_epoch 30 \
                                      --batch_size 1 \
                                      --loss nll_surv_l1 \
                                      --lr 0.001 \
                                      --optimizer SGD \
                                      --scheduler None \
                                      --alpha 100.0

CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv \
                                      --dataset tcga_gbmlgg \
                                      --data_root_dir /master/zhou_feng_tao/data/TCGA/GBMLGG/x20 \
                                      --modal coattn \
                                      --model cmta \
                                      --num_epoch 30 \
                                      --batch_size 1 \
                                      --loss nll_surv_l1 \
                                      --lr 0.001 \
                                      --optimizer SGD \
                                      --scheduler None \
                                      --alpha 1.0

CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv \
                                      --dataset tcga_luad \
                                      --data_root_dir /master/zhou_feng_tao/data/TCGA/LUAD/x20 \
                                      --modal coattn \
                                      --model cmta \
                                      --num_epoch 30 \
                                      --batch_size 1 \
                                      --loss nll_surv_l1 \
                                      --lr 0.001 \
                                      --optimizer SGD \
                                      --scheduler None \
                                      --alpha 0.0001

CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv \
                                      --dataset tcga_ucec \
                                      --data_root_dir /master/zhou_feng_tao/data/TCGA/UCEC/x20 \
                                      --modal coattn \
                                      --model cmta \
                                      --num_epoch 30 \
                                      --batch_size 1 \
                                      --loss nll_surv_l1 \
                                      --lr 0.001 \
                                      --optimizer SGD \
                                      --scheduler None \
                                      --alpha 1.0