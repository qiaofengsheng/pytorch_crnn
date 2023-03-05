## 训练脚本
    python -m torch.distributed.launch --nproc_per_node=2 train.py
## 推理脚本
    python -B inference