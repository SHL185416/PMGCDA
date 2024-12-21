# Prototype-based Multi-source Graph Contrastive Domain Adaptation (PMGCDA)

![image-20241221204720998](https://gitee.com/l18541900/picgo/raw/master/img/202412212123013.png)

This repository provides the Pytorch code for the work "Prototype-based Multi-source Graph Contrastive Domain Adaptation" published in XXX, 202X.



We propose a Prototype-based Multi-source Graph Contrastive Domain Adaptation (PMGCDA) model. The proposed PMGCDA model aligns the source and target domains based on local and global prototypes. Local prototype-based graph contrastive domain adaptation aligns the same class of nodes across different graphs to enhance intra-class consistency. While global prototype-based graph contrastive domain adaptation pushes apart the prototypes of different classes to reduce class confusion. In addition, the node-level and domain-level transferability weights are introduced to dynamically control the impact of different source graphs on each individual target node and on the whole target graph, respectively. 



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
python main.py --Clf_wei=1 --P_wei=1 --Prot_wei=0.01 --batch_size=8000 --data_key=citation --epochs=100 --gpu=0 --in_drop=0.4 --l2_w=0.01 --lr_ini=0.01 --num_hidden=64 --target=citation1_acmv9 --tau_p=0.5 --random_number=2024
```

For more details of this multi-source domain adaptation approach, please refer to the following work:

@article{XXX,
title = {Prototype-based Multi-source Graph Contrastive Domain Adaptation},
journal = {},
volume = {},
pages = {},
year = {202X},
url = {https://www.},
author = {}
}

If you have any questions regarding the code, please contact email [cylin@hainanu.edu.cn](mailto:cylin@hainanu.edu.cn).