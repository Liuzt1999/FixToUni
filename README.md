# MultiMatch
A semi-supervised method with layer consistency and ranking consistency  
## Requirements

- Python3.6+
- torch 1.4
- torchvision 0.5
- tqdm
- tensorboard
- apex (optional)

## Train
Train the model by 4000 labeled data of CIFAR-10 dataset:  

`python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/cifar10@4000.5 --ranking_way triplet --gen_anchor CE`
