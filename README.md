# 花卉小能手

## II-初识卷积神经网络

- 一张RGB三通道彩色图片是如何被处理成可用于训练的数据？张量又是什么？

将一张RGB三通道图片转换为一个三维的张量，其中三个维度分别为：宽、高、RGB通道，然后通过卷积提取其中的特征进而转换成模型可用的数据

**张量**是一种多维数组，是深度学习框架中表示和计算数据的基本单位

- 一个卷积神经网络分为哪几层？

输入层、卷积层、激活层、池化层、全连接层、输出层

- 什么是卷积，数学严格意义上的卷积和神经网络中的卷积操作有区别吗？卷积核、步长是什么？

~~（我数学不好喵）~~

卷积是用一个卷积核在各个部分进行加权求和，从而提取局部特征

神经网络中的卷积与数学严格意义上的卷积是有区别的，询问ai后其给我的答案是：数学中的卷积其包含一个翻转核的过程，但在神经网络中实际使用的是互相关，相比数学上没有翻转核的过程

**卷积核**是一个张量，内部存有各种权重参数，可以在输入特征上滑动，并在该过程中提取特征

**步长**则是控制卷积核在特征图或者池化窗口上每次移动的长度

- 池化有哪些类型，其作用分别有什么？感受野是什么？

**池化**是在局部区域内用一个统计量代替该区域（一般不减少通道数），从而减小特征图的空间尺寸，常见池化类型有：

1. 最大池化（取窗口内最大值）

作用：保留最显著特征

2. 平均池化（取窗口内的平均值）

作用：平滑特征

3. 求和池化（对窗口内值进行求和）

作用：聚合局部特征

4. ……

**感受野**是输出特征图上的一个像素（或神经元），对应到原始输入图像上能的区域大小

池化可以扩大感受野

- 激活层在此的作用？最后卷积神经网络会输出什么？

激活层在此的作用和之前神经网络一样，为引入非线性，若不加入激活层，多层神经网络堆叠起来仍只等价于一个线性变换

对于图像分类任务，最终卷积神经网络会输出每个种类的概率

## III-数据预处理

- 数据集中存在异常或者损坏的图片无法打开（图片默认格式为jpg）；

把训练数据交给ai看了以后，发现数据存在以下问题：

1. 有7个0字节空文件
2. 有6个28字节的假图片
3. 有图片是完全相同的文件，却挂在两个不同类别下
4. 有测试集里的图片原封不动地出现在训练集里

处理方法如下：

**对于空文件与假图片**

通过识别128字节以下的图片文件找出假图片和空文件

将其移动至隔离文件夹

（中间报错了许多次，重置了多次数据以测试，具体情况包括但不限于：

因为运行两次导致第二次扫描了隔离文件夹导致报错，后续添加了跳过隔离文件夹以解决

错误文件中包含同名文件导致移动文件时报错，后续通过在隔离文件夹中保留其根目录以解决，同时产生下面的衍生错误

保留根目录时忘记在前面加上隔离文件夹的路径，导致根目录被创建在文件夹`data`之外，导致后续移动找不到根目录报错，加上后解决

我是小祝QAQ）

**对于相同文件**

通过python将其找出，然后通过人工识别删除错误的部分（其实占比小好像是可以全删了的）

我写了一个通过哈希辨别相同文件的函数`scan_duplicates()`

在`main.py`中运行`scan_duplicates("data/train")`后输出如下

```
[[WindowsPath('data/train/rose/15922772266_1167a06620.jpg'), WindowsPath('data/train/tulip/15922772266_1167a06620.jpg')], [WindowsPath('data/train/rose/17104364030_ee31ee279b_n.jpg'), WindowsPath('data/train/tulip/17104364030_ee31ee279b_n.jpg')]]
```

感觉直接靠名称锁定都可以

15922772266_1167a06620.jpg应该是玫瑰

17104364030_ee31ee279b_n.jpg应该也是玫瑰

手动删掉tulip下的同名文件

再次运行`scan_duplicates("data/train")`发现输出变为空列表，表明去重成功

运行`scan_duplicates("data/test")`发现测试集中没有相同文件

**对于训练集与测试集中的相同文件**

运行`scan_duplicates("data")`

```
[[WindowsPath('data/train/rose/17291930305_79deae6a90_n.jpg'), WindowsPath('data/test/tulip/17291930305_79deae6a90_n.jpg')], [WindowsPath('data/train/sunflower/14889392928_9742aed45b_m.jpg'), WindowsPath('data/test/sunflower/15066430311_fb57fa92b0_m.jpg')], [WindowsPath('data/train/sunflower/15069459615_7e0fd61914_n.jpg'), WindowsPath('data/test/sunflower/15072973261_73e2912ef2_n.jpg')], [WindowsPath('data/train/tulip/17291451621_0e39f08b9e_n.jpg'), WindowsPath('data/test/rose/17291451621_0e39f08b9e_n.jpg')], [WindowsPath('data/quarantine/train/daisy/11694430126_056ffea63c_n.png'), WindowsPath('data/quarantine/train/daisy/34694430126_056ffea63c_n.png'), WindowsPath('data/quarantine/train/dandelion/1069454126_056ffea63c_n.png'), WindowsPath('data/quarantine/train/dandelion/20694430126_056ffea63c_n.png'), WindowsPath('data/quarantine/train/dandelion/31658430126_056ffea63c_n.jpg'), WindowsPath('data/quarantine/train/rose/13658454126_056ffea63c_n.jpg'), WindowsPath('data/quarantine/train/rose/31658430126_056ffea63c_n.jpg')], [WindowsPath('data/quarantine/train/sunflower/1214997_8576bf4ea3.png'), WindowsPath('data/quarantine/train/sunflower/6314997_8576bf4ea3.png'), WindowsPath('data/quarantine/train/sunflower/6514952_8576bf4ea3.png'), WindowsPath('data/quarantine/train/tulip/2098797_8576bf4ea3.png'), WindowsPath('data/quarantine/train/tulip/21214940_8576bf4ea3.png'), WindowsPath('data/quarantine/train/tulip/4414997_8576bf4ea3.png')]]
```

~~草忘记将隔离区除外了~~

加入隔离区除外后再次运行

```
[[WindowsPath('data/train/rose/17291930305_79deae6a90_n.jpg'), WindowsPath('data/test/tulip/17291930305_79deae6a90_n.jpg')], [WindowsPath('data/train/sunflower/14889392928_9742aed45b_m.jpg'), WindowsPath('data/test/sunflower/15066430311_fb57fa92b0_m.jpg')], [WindowsPath('data/train/sunflower/15069459615_7e0fd61914_n.jpg'), WindowsPath('data/test/sunflower/15072973261_73e2912ef2_n.jpg')], [WindowsPath('data/train/tulip/17291451621_0e39f08b9e_n.jpg'), WindowsPath('data/test/rose/17291451621_0e39f08b9e_n.jpg')]]
```

~~这次文件名不同了，还好我是靠哈希检测的~~

解决方法

将test里重复的部分删去（手动），尽量避免训练数据减少

再次运行，输出空列表，说明有效

以上相关代码见`src/data_clean.py`

- 划分数据集；

依旧使用我最爱的for循环遍历文件，然后通过random随机取对应个数索引值划分训练集和验证集

以上相关代码见`src/split.py`

- 进行数据增强，扩充样本容量（可调库）；

我对数据增强了解不多，此处ai辅助使用较多

~~可恶数据增强相关函数怎么那么多😭😭😭~~

以上相关代码见`src/data_augment.py`

- 调整图片大小以适应CNN的输入；

好像数据增强里ai已经帮我做了，即transforms.Resize(SIZE)

## IV-炼丹

1. 这一部分直接涉及到代码的实现与模型的训练，我们有必要再去深入了解编程中可能用到的各种概念与工具。请自主学习并回答下面的概念问题：

~~这些好像在一轮时问过别~~

- 什么是超参数？如何设定？

超参数与参数不同，一般由训练模型的人设置，而不由模型自动更新

常见的超参数有学习率，归一化参数等

我不知道怎么描述如何设定~~语言表达能力宛若成年香蕉😭😭~~

大概就是诸如

```Python
LR=
SIZE=
EPOCHS=

……
```

的各种赋值

- 什么是优化器，有哪些类型？

优化器可以根据梯度更改各种参数

类型包括：

Adam、AdamW、AdaGrad、RMSProp等

- DataLoader 与 DataSet是什么及其使用？（可调库）

Dataset是PyTorch使用的数据集，Dataloader则决定怎么读取数据

直接看代码吧（

2. 依据图片中的网络结构补全代码，完成网络的设计：

~~这个好难算啊😭😭~~

看代码吧

3. 图片存在docs里

损失函数：`nn.CrossEntropyLoss()`
优化器：AdamW
EPOCHS：60
种子均为325

~~其实一开始EPOCHS是250，但是发现训练一次要一个小时于是放弃了~~

## V-模型的优化与微调（选做）

1. 相信大家有疑问，我如果一直堆叠网络的层数，岂不是能无限的提高模型的能力？模型层数越多、结构越深，性能就一定越好吗？请自行查阅资料学习。

这个查了，层数太多可能导致参数量过大，若数据量不变，模型会记下所有的数据，导致过拟合

因此，并非层数越多越好，而是要选择适合的层数

2. 学习现代新的网络结构(如ResNet、VGG)，并尝试利用其训练模型，对比两个模型的能力。

这个没做喵

3. 模型训练好了该如何使用呢？难道每次使用前都需要重新训练一次？请学习模型的保存与载入方法，将自己的模型保存在本地，自己在网上找到图片后丢给模型分类。

模型打包的方法我问ai了，模型也已经保存在`src/flower.pt`了（现在发现太大传不上GitHub，因此需要再次运行生成）

大致就是通过`torch.save()`将一个存储在字典中的模型数据存下来

读取的方法就是建一个类，将刚刚输出的文件传给它，让其加载原本模型训练出的数据就可以使用了

网上随便找了两张图，在`demo.py`里导入main，并加载flower.pt后，可以正常使用，将图交给模型后输出如下

```
设备：cuda
3734.jpg_wh860.jpg是rose
33331825_155457689128_2.jpg是tulip
```

终于写完了😭😭😭
