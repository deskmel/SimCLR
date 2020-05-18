#### 简介
这个SimCLR代码实现是基于 https://github.com/sthalles/SimCLR 实现的
除了论文使用的NT-Xent Loss 之外，添加了两种论文中讨论的 Contrastive Loss 分别是NT-logistic Loss 和 Marginal Triplet Loss
Loss | 数学表达形式
- | -
NT-Xent |$u^Tv^+/\tau-log\sum_{v\in\{v^+,v^-\}}exp(u^Tv/\tau)$
NT-Logistic |$log\sigma(u^Tv^+/\tau)+log\sigma(-u^Tv^-/\tau)$
Marginal Triplet |$-max(u^Tv^--u^Tv^++m,0)$

并在CIFAR10 数据集上做了对比实验
下面给出每种loss最好的实验结果

 Loss|Resnet | Feature demension | batchsize | epoch | $\tau$ / m|CIFAR10 ACC|
-|-|-|-|-|-|-
nt_xent|resnet50|128|128|100|0.5|0.8387
nt_logistic|resnet50|128|128|100|0.5|0.8094
marginal_triplet|resnet50|128|128|100|1|0.8100


#### 代码目录
个人实现的代码主要包括
两个Loss函数
```
./loss/nt_logistic.py
./loss/marginal_triplet.py
```
CIFAR10数据集预处理和加载
```
./data_aug/data_wrapper.py
```
基于原论文的形容修改resnet50的结构
```
./model/resnet_simclr.py
```
模型的评估文件
```
./eval.py
```
#### 实验过程和一些个人理解
由于这是我第一次接触深度学习相关的无监督学习内容，因此可能对某些模块理解的不是特别深刻。

##### Loss 相关实现
首先simclr训练时，每个样本会产生一对正样本，而输入时同一个batch里的其他样本都会被当作负样本。
###### Marginal Triplet Loss
上文中提到的 marginal triplet loss 表达的形式是对于一对正负样本，其含义便是期望输入样本和正样本的相似度减去和负样本的相似度可以大于阈值m值。
扩展到本文中的情况便是,对于每一个样本，他有一个对应的正样本和$2*(batchsize-1)$个负样本,对这个样本每一个负样本我们重复使用同一个正样本计算marginal triplet loss。首先定义
$$
l(i,j) = \frac{1}{2*(N-1)}\sum_{k=1}^{2N} 1_{(k\neq i,j)} max(s_{i,k}-s_{i,j}+m,0)
$$
总的Loss便为
$$L = \frac{1}{2N}\sum_{i}^{N}[l(2i,2i+1)+l(2i+1,2i)]$$

###### NT_Logistic
NT_logistic 可以理解为一种逻辑回归在这里的扩展版本。对于每个样本，他存在一个对应的正样本和$2(N-1)$个负样本。若把 $\sigma(s_{i,j}/\tau)$视为样本$i,j$为相似样本的可能性，这个loss的目标即为获得最大似然。 因此定义
$$
l(i,j) = \left\{ 
\begin{aligned}
 & log(\sigma(s_{i,j}/\tau)) & if (i=j-1,j=i-1)\\
 & log(\sigma(-s_{i,j}/\tau)) & otherwise\\
\end{aligned}
\right.
$$
不过一开始个人只是简单将所有对数似然损失加起来，并没有考虑样本数量的不对称性，因此一开始的实现为
$$L = \frac{1}{2N*(2N-1)}\sum_{i=1}^{2N}\sum_{j=1}^{2N}1_{(j\neq i)}(l(i,j)) $$
即对每一个样本计算他和其他样本之间的对数似然误差

但是这样训练的结果存在比较大的问题，即逻辑回归本质类似一个线性分类器，在正负样本有强烈不均的情况下，训练结果会有较大误差

因此在考虑了转发样本数量的情况下，将loss修改为
$$
L = \frac{1}{4N*(N-1)}\sum_{i=1}^{N} (4(N-1)l(2i,2i+1)+\sum_{j=1}^{2N}1_{(j\neq 2i,j\neq 2i+1)}(l(2i,j)+l(2i+1,j)))
$$
事实上相当于扩展正样本数量，使得loss计算时，重复计算正样本的对数似然误差至其和负样本数量一致。
##### 基于cifar10数据集修改网络模型结果和数据预处理
第一次模型基于三种loss的结果如下
 Loss|Resnet | Feature demension | batchsize | epoch | $\tau$ / m|CIFAR10 ACC|
-|-|-|-|-|-|-
nt_xent|resnet50|256|512|100|0.5|0.5701
nt_logistic|resnet50|256|512|100|0.5|0.3314
marginal_triplet|resnet50|256|512|100|1|0.5329


结果并不理想。 其中nt_logistic loss的结果特别差，甚至不如pca聚类结果，原因已经在上文中分析。
而其他结果不理想的原因在论文中得到解答，由于cifar10数据集输入图片大小（32，32）比较小，resnet50第一个7*7的conv和maxpool严重的削弱了他的特征表达能力。因此论文附录中提到要修改第一个conv为3*3 stride为1并删去 maxpool1，此外在数据预处理的阶段还去除了高斯模糊变换，并设置颜色变换的力度为0.5。
因此个人重新修改了模型和数据预处理，重新训练。
此次的结果为
 Loss|Resnet | Feature demension | batchsize | epoch | $\tau$ / m|CIFAR10 ACC|
-|-|-|-|-|-|-
nt_xent|resnet50|128|128|100|0.5|0.8387
nt_logistic|resnet50|128|128|100|0.5|0.8094
marginal_triplet|resnet50|128|128|100|1|0.8100


##### 参数搜索的结果
 Loss|Resnet | Feature demension | batchsize | epoch | $\tau$ / m|CIFAR10 ACC|
-|-|-|-|-|-|-
nt_xent|resnet50|128|128|100|0.1|0.8387
nt_xent|resnet50|128|128|100|0.5|0.8387
nt_xent|resnet50|128|128|100|1|0.8387
nt_logistic|resnet50|128|128|100|0.1|0.8094
nt_logistic|resnet50|128|128|100|0.5|0.8094
nt_logistic|resnet50|128|128|100|1|0.8094
marginal_triplet|resnet50|128|128|100|0|0.8100
marginal_triplet|resnet50|128|128|100|0.5|0.8100
marginal_triplet|resnet50|128|128|100|1|0.8100

##### 对Loss的一些讨论
事实上可以看出来在实验对比下nt_xent 毫无疑问更有竞争力，分析原因