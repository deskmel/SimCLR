#### 简介
这个SimCLR代码是基于 https://github.com/sthalles/SimCLR 实现的

在此基础上，添加了两种论文中讨论的 Contrastive Loss : NT-logistic Loss 和 Marginal Triplet Loss

并对这三种Loss 函数在CIFAR10 数据集上做了对比实验

下面给出每种loss最好的实验结果

 Loss|Resnet | Feature demension | batchsize | epoch | temperature / m|CIFAR10 ACC|
-|-|-|-|-|-|-
nt_xent|resnet50|128|128|100|0.5|0.8387
nt_logistic|resnet50|128|128|100|0.5|0.8094
marginal_triplet|resnet50|128|128|100|1|0.8100

##### 三种Loss 和他的数学表达形式

###### NT-Xent  

![](https://latex.codecogs.com/gif.latex?u^Tv^+/\tau-log\sum_{v\in\{v^+,v^-\}}exp(u^Tv/\tau))

###### NT-Logistic 

![](https://latex.codecogs.com/gif.latex?log\sigma(u^Tv^+/\tau)+log\sigma(-u^Tv^-/\tau))

###### Marginal Triplet 

![](https://latex.codecogs.com/gif.latex?-max(u^Tv^--u^Tv^++m,0))






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
##### Loss 相关实现
首先simclr训练时，每个样本会产生一对正样本，而输入时同一个batch里的其他样本都会被当作负样本。Loss的目的往往是训练一个线性分类器，将某样本对应的正样本和负样本进行区分。
###### Marginal Triplet Loss
上文中提到的 marginal triplet loss 表达的形式是对于一对正负样本求loss。其含义便是期望输入样本和正样本的相似度减去和负样本的相似度可以大于阈值m值。

扩展到本文中的情况便是,对于每一个样本，他有一个对应的正样本和2*(batchsize-1)个负样本,对这个样本每一个负样本我们重复使用同一个正样本计算marginal triplet loss。

首先定义

![](https://latex.codecogs.com/svg.latex?l(i,j)%20=%20\frac{1}{2*(N-1)}\sum_{k=1}^{2N}%201_{(k\neq%20i,j)}%20max(s_{i,k}-s_{i,j}+m,0))


总的Loss便为

![](https://latex.codecogs.com/svg.latex?L%20=%20\frac{1}{2N}\sum_{i}^{N}[l(2i,2i+1)+l(2i+1,2i)])

###### NT_Logistic
NT_logistic 可以理解为一种逻辑回归在这里的扩展版本。对于每个样本，他存在一个对应的正样本和2(N-1)个负样本。若把 ![](https://latex.codecogs.com/svg.latex?\sigma(s_{i,j}/\tau))视为样本i,j为相似样本的可能性，这个loss的目标即为获得最大似然估计。 

因此定义其对数似然损失为

![](https://latex.codecogs.com/gif.latex?l%28i%2Cj%29%20%3D%20%5Cleft%5C%7B%20%5Cbegin%7Baligned%7D%20%26%20log%28%5Csigma%28s_%7Bi%2Cj%7D/%5Ctau%29%29%20%26%20if%20%28i%3Dj-1%2Cj%3Di-1%29%5C%5C%20%26%20log%28%5Csigma%28-s_%7Bi%2Cj%7D/%5Ctau%29%29%20%26%20otherwise%5C%5C%20%5Cend%7Baligned%7D%20%5Cright.)

不过在开始阶段个人只是简单将所有对数似然损失加起来，并没有考虑样本数量的不对称性，因此最初的的实现版本为

![](https://latex.codecogs.com/svg.latex?L%20=%20\frac{1}{2N*(2N-1)}\sum_{i=1}^{2N}\sum_{j=1}^{2N}1_{(j\neq%20i)}(l(i,j)))

即对每一个样本计算他和其他样本之间的对数似然误差

但是这样训练的结果存在比较大的问题，即逻辑回归本质类似一个线性分类器，在正负样本有强烈不均的情况下，训练结果会有较大误差

因此在考虑了转发样本数量的情况下，将loss修改为

![](https://latex.codecogs.com/svg.latex?L%20=%20\frac{1}{4N(N-1)}\sum_{i=1}^{N}%20(4(N-1)l(2i,2i+1)+\sum_{j=1}^{2N}1_{(j\neq%202i,j\neq%202i+1)}(l(2i,j)+l(2i+1,j))))

这事实上相当于扩展正样本数量，使得loss计算时，重复计算正样本的对数似然误差至其和负样本数量一致。

##### 基于cifar10数据集修改网络模型结果和数据预处理
第一次模型基于三种loss(nt_logistic的实现为最初版本）的结果如下

 Loss|Resnet | Feature demension | batchsize | epoch | t / m|CIFAR10 ACC|
-|-|-|-|-|-|-
nt_xent|resnet50|256|512|100|0.5|0.5701
nt_logistic|resnet50|256|512|100|0.5|0.3314
marginal_triplet|resnet50|256|512|100|1|0.5329

结果并不理想。 其中nt_logistic loss的结果特别差，甚至不如pca聚类结果，原因已经在上文中分析，便是未考虑正负样本不均的情况。

而其他结果也不理想的原因在论文中得到解答：由于cifar10数据集输入图片大小（32，32）比较小，resnet50第一个7\*7的卷积层和池化层严重的削弱了他的特征表达能力。因此论文附录中提到要修改第一个卷积层为3\*3 步长为1 并删去池化层。此外论文还提到要在数据预处理的阶段去除了高斯模糊变换，并设置颜色变换的力度为0.5。

因此个人重新修改了模型和数据预处理响应的代码，重新训练。

此次的结果为
 Loss|Resnet | Feature demension | batchsize | epoch | t / m|CIFAR10 ACC|
-|-|-|-|-|-|-
nt_xent|resnet50|128|128|100|0.5|0.8387
nt_logistic|resnet50|128|128|100|0.5|0.8094
marginal_triplet|resnet50|128|128|100|1|0.8100

##### 参数搜索的结果
 Loss|Resnet | Feature demension | batchsize | epoch | t / m|CIFAR10 ACC|
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