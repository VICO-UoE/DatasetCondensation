
## Dataset Condensation with Gradient Matching [[PDF]](https://openreview.net/pdf?id=mSAKhLYLSsl)


### Method
<p align="center"><img src='docs/method.png' width=750></p>
<center>Figure 1: Dataset Condensation (left) aims to generate a small set of synthetic images that can match the performance of a network trained on a large image dataset. Our method (right) realizes this goal by learning a synthetic set such that a deep network trained on it and the large set produces similar gradients w.r.t. the parameters. The synthetic data can later be used to train a network from scratch in a fraction of the original computational load. CE denotes Cross-Entropy. </center><br>

### Setup
install packages in the requirements.

###  Basic experiments - Table 1
```
python main.py  --dataset CIFAR10  --model ConvNet  --ipc 10
# --dataset: MNIST, FashionMNIST, SVHN, CIFAR10
# --ipc (images/class): 1, 10, 20, 30, 40, 50
```

### Cross-architecture experiments - Table 2
```
python main.py  --dataset MNIST  --model ConvNet  --ipc 1  --eval_mode M
# --model: MLP, LeNet, ConvNet, AlexNet, VGG11BN, ResNet18BN_AP, Note: set --lr_img 0.01 when --model MLP
```

### Ablation study on different modules - Table T2, T3, T4, T5, T6, T7
```
python main.py  --dataset MNIST  --model ConvNetW32  --eval_mode W  --ipc 1 
python main.py  --dataset MNIST  --model ConvNetW64  --eval_mode W  --ipc 1
python main.py  --dataset MNIST  --model ConvNetW128  --eval_mode W  --ipc 1
python main.py  --dataset MNIST  --model ConvNetW256  --eval_mode W  --ipc 1

python main.py  --dataset MNIST  --model ConvNetD1  --eval_mode D  --ipc 1
python main.py  --dataset MNIST  --model ConvNetD2  --eval_mode D  --ipc 1
python main.py  --dataset MNIST  --model ConvNetD3  --eval_mode D  --ipc 1
python main.py  --dataset MNIST  --model ConvNetD4  --eval_mode D  --ipc 1

python main.py  --dataset MNIST  --model ConvNetAS  --eval_mode A  --ipc 1
python main.py  --dataset MNIST  --model ConvNetAR  --eval_mode A  --ipc 1
python main.py  --dataset MNIST  --model ConvNetAL  --eval_mode A  --ipc 1

python main.py  --dataset MNIST  --model ConvNetNP  --eval_mode P  --ipc 1
python main.py  --dataset MNIST  --model ConvNetMP  --eval_mode P  --ipc 1 
python main.py  --dataset MNIST  --model ConvNetAP  --eval_mode P  --ipc 1

python main.py  --dataset MNIST  --model ConvNetNN  --eval_mode N  --ipc 1
python main.py  --dataset MNIST  --model ConvNetBN  --eval_mode N  --ipc 1 
python main.py  --dataset MNIST  --model ConvNetLN  --eval_mode N  --ipc 1
python main.py  --dataset MNIST  --model ConvNetIN  --eval_mode N  --ipc 1
python main.py  --dataset MNIST  --model ConvNetGN  --eval_mode N  --ipc 1


python main.py  --dataset MNIST  --model ConvNet  --ipc 1  --dis_metric mse
# --dis_metric (gradient distance metrics): ours, mse, cos
# --model: MLP, LeNet, ConvNet, AlexNet, VGG11BN, ResNet18BN_AP
```


### Performance
|  | MNIST | FashionMNIST | SVHN | CIFAR10 |
 :-: | :-: | :-: | :-: | :-:
| 1 img/cls  | 91.7 | 70.5 | 31.2 | 28.3 |
| 10 img/cls | 97.4 | 82.3 | 76.1 | 44.9 |
| 50 img/cls | 98.8 | 83.6 | 82.3 | 53.9 |

Table 1: Testing accuracies (%) of ConvNets trained from scratch on 1, 10 or 50 synthetic image(s)/class.


### Visualization
<p align="center"><img src='docs/1ipc.png' width=500></p>
<center>Figure 2: Visualization of condensed 1 image/class with ConvNet for MNIST, FashionMNIST, SVHN and CIFAR10. Average testing accuracies on randomly initialized ConvNets are 91.7%, 70.5%, 31.2% and 28.3% respectively. </center><br>


<p align="center"><img src='docs/10ipc.png' width=800></p>
<center>Figure 3: Visualization of condensed 10 images/class with ConvNet for MNIST, FashionMNIST, SVHN and CIFAR10. Average testing accuracies on randomly initialized ConvNets are 97.4%, 82.3%, 76.1% and 44.9% respectively. </center><br>



### Citation
```
@inproceedings{
zhao2021dataset,
title={Dataset Condensation with Gradient Matching},
author={Bo Zhao and Konda Reddy Mopuri and Hakan Bilen},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=mSAKhLYLSsl}
}
```

