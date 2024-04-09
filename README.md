# MultiMatch
A semi-supervised method with layer consistency and ranking consistency  

## Installation
- conda create -n MultiMatch python=3.10.4
- conda activate MultiMatch
- pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
- pip install tensorboard
- pip install tqdm
- pip install apex

## Train
Train the model by 4000 labeled data of CIFAR-10 dataset:  

`python train1.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/cifar10@4000.5 --ranking_way triplet --gen_anchor CE`  
