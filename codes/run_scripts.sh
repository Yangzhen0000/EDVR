# single GPU training (image SR)
# python train.py -opt options/train/train_SRResNet.yml
# python train.py -opt options/train/train_SRGAN.yml
# python train.py -opt options/train/train_ESRGAN.yml


# distributed training (video SR)
# 8 GPUs
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py -opt options/train/train_EDVR_woTSA_M.yml --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py -opt options/train/train_EDVR_M.yml --launcher pytorch

# distributed training (video BDE)
# 2 GPUs
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 train.py -opt options/train/train_EDVR_BDE.yml --launcher pytorch
# 1 GPU
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch train.py -opt options/train/train_EDVR_BDE.yml --launcher pytorch
