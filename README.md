# CCA-AGC
Implementation for CCA-AGC model (A Contrastive Learning Method with Cluster-preserving Augmentation for Attributed Graph Clustering).

# Implementation
pretrain.py: pretrain multilevel contrast to get initial parameters and node representations.

train_conclu.py: jointly train the whole model.

#### Example:

```
python train_conclu.py --dataset Cora --hidden 512 --out_dim 256 --pro_hid 1024 --activation relu --k 0 --rm 85 --mask 0.1 --lr 0.0001 --rep 0.1
```

