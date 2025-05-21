# Prototypical Multi-source Graph Contrastive Domain Adaptation (PMGCDA)

![image-20250507224137660](https://gitee.com/l18541900/picgo/raw/master/img/202505072241329.png)

This repository provides the Pytorch code for the work "Prototypical Multi-source Graph Contrastive Domain Adaptation" published in XXX, 202X.

#### Environment requirement

All experiments were conducted on a system equipped with dual RTX 3080 GPUs (20GB each), a 12-core Intel Xeon Platinum 8352V CPU @ 2.10GHz, and 48GB of RAM.

The code has been tested running under the required packages as follows:

- torch==1.13.0+cu116
- dgl==0.6.1
- numpy==1.24.4
- scipy==1.8.1
- scikit-learn==1.1.1

#### Dataset folder

The folder structure required

- input
  - citation
    - `citation1_acmv9.mat`
    - `citation1_citationv1.mat`
    - `citation1_dblpv7.mat`
    - `citation2_acmv8.mat`
    - `citation2_citationv1.mat`
    - `citation2_dblpv4.mat`

##### How to run

```shell
python main.py  --Clf_wei=10 --P_wei=1 --Prot_wei=10 --batch_size=8000 --data_key=citation --epochs=100 --gpu=0 --in_drop=0.3 --l2_w=0.01 --lr_ini=0.01 --num_hidden=64 --target=citation1_citationv1 --tau_p=0.5 --random_number=2024
```

For more details of this multi-source domain adaptation approach, please refer to the following work:

@article{XXX,
title = {Prototypical Multi-source Graph Contrastive Domain Adaptation},
journal = {},
volume = {},
pages = {},
year = {202X},
url = {https://www.},
author = {}
}

If you have any questions regarding the code, please contact email [cylin@hainanu.edu.cn](mailto:cylin@hainanu.edu.cn).
