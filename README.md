# 动手学深度学习 李沐 dive-into-deep-learning

李沐老师的课程中源码都是用jupyter notebook写的；这里全部使用pycharm编辑器来编程，改写为py格式。  
希望可以记录课程的学习过程，同时能帮助他人。

### 课程相关资料
1. 课程的直播地址：http://courses.d2l.ai/zh-v2/
2. 课程的课件地址：https://zh-v2.d2l.ai/
3. 另一个可参考的笔记：https://tangshusen.me/Dive-into-DL-PyTorch

### 本笔记的目录
##### ch01. 预备知识  
1.1. [数据操作](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch01/01-ndarray.py)  
1.2. [数据预处理](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch01/02-pandas.py)  
1.3. [线性代数](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch01/03-linear-algebra.py)  
1.4. [微分](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch01/04-calculus.py)  
1.5. [自动求导](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch01/05-autograd.py)  
##### ch02. 线性神经网络  
2.1. [线性回归](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch02/01-linear-regression.py)  
2.2. [线性回归的从零开始实现](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch02/02-linear-regression-scratch.py)  
2.3. 线性回归的简洁实现  
2.4. softmax回归  
2.5. [图像分类数据集](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/d2lutil/common.py)  
2.6. [softmax回归的从零开始实现](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch02/03-softmax-linear-regression-scratch.py)  
2.7. [softmax回归的简洁实现](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch02/04-softmax-linear-regression-concise.py)  
##### ch03. 多层感知机  
3.1. [多层感知机](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch03/01-mlp.py)  
3.2. [多层感知机的从零开始实现](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch03/02-mlp-from-zero.py)  
3.3. [多层感知机的简洁实现](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch03/03-mlp-simple.py)  
3.4. [模型选择、欠拟合和过拟合](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch03/04-underfit-overfit.py)  
3.5. [权重衰减](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch03/05-weight-decay-simple.py)  
3.6. [Dropout](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch03/06-dropout-simple.py)  
3.7. 正向传播、反向传播和计算图  
3.8. 数值稳定性和模型初始化  
3.9. 环境和分布偏移  
3.10. [实战 Kaggle 比赛：预测房价](https://github.com/Miraclelucy/dive_into_deep_learning/blob/main/ch03/10-kaggle-house-price.py)   
##### ch04. 深度学习计算  
4.1. [层和块](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch04/01-model-construction.py)  
4.2. [参数管理](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch04/02-parameters.py)  
4.3. 延后初始化  
4.4. [自定义层](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch04/03-custom-layer.py)  
4.5. [读写文件](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch04/04-read-write.py)  
4.6. GPU  
##### ch05. 卷积神经网络   
5.2. [图像卷积](https://github.com/Miraclelucy/dive_into_deep_learning/blob/main/ch05/02-conv-layer.py)      
5.3. [填充和步幅](https://github.com/Miraclelucy/dive_into_deep_learning/blob/main/ch05/03-padding-and-strides.py)   
5.4. [多输入多输出通道](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch05/04-channels.py)    
5.5. [汇聚层](https://github.com/Miraclelucy/dive-into-deep-learning/blob/main/ch05/05-pooling.py)  
5.6. 卷积神经网络（LeNet） 
##### ch06. 现代卷积神经网络  
6.1. 深度卷积神经网络（AlexNet）  
6.2. 使用重复元素的网络（VGG）  
6.3. 网络中的网络（NiN）  
6.4. 含并行连结的网络（GoogLeNet）  
6.5. 批量归一化  
6.6. 残差网络（ResNet）  
6.7. 稠密连接网络（DenseNet） 
##### ch07.  循环神经网络
7.1. 序列模型  
7.2. 文本预处理 
7.3. 语言模型和数据集（周杰伦专辑歌词）  
7.4. 循环神经网络  
7.5. 循环神经网络的从零开始实现  
7.6. 循环神经网络的简洁实现  
7.7. 通过时间反向传播  
##### ch08.  现代循环神经网络  
8.1. 门控循环单元（GRU）    
8.2. 长短期记忆（LSTM）   参考看下这个 https://colah.github.io/posts/2015-08-Understanding-LSTMs/    
8.3. 深度循环神经网络    
8.4. 双向循环神经网络    
8.5. 机器翻译与数据集    
8.6. 编码器-解码器结构  
8.7. 序列到序列学习  
8.8. 束搜索  
##### ch09.  注意力机制  
9.1. 注意力机制    
9.2. 注意力汇聚    
9.3. 注意力评分函数   
9.4. Bahdanau注意力    
9.5. 多头注意力  
9.6. 自注意力和位置编码   
9.7. Transformer  
##### ch10.  优化算法  
10.1. 优化和深度学习  
10.2. 凸性  
10.3. 梯度下降  
10.4. 随机梯度下降  
10.5. 小批量随机梯度下降  
10.6. 动量法    
10.7. AdaGrad算法  
10.8. RMSProp算法  
10.9. Adadelta算法  
10.10. Adam算法  
10.11. 学习率调度器  
##### ch11.  计算性能
11.1. 编译器和解释器   
11.2. 异步计算  
11.3. 自动并行  
11.4. 硬件  
11.5. 多GPU训练  
11.6. 多GPU的简洁实现  
11.7. 参数服务器  
##### ch12.  计算机视觉
12.1. 图像增广  
12.2. 微调  
12.3. 目标检测和边界框  
12.4. 锚框   
12.5. 多尺度目标检测  
12.6. 目标检测数据集（皮卡丘）   
12.7. 单发多框检测（SSD）  
12.8. 区域卷积神经网络（R-CNN）系列  
12.9. 语义分割和数据集  
12.10. 转置卷积  
12.11. 全卷积网络  
12.12. 风格迁移 
##### ch13.  自然语言处理：预训练
13.1. 词嵌入  
13.2. 近似训练  
13.3. 用于预训练词嵌入的数据集  
13.4. 预训练word2vec  
13.5. 全局向量的词嵌入  
13.6. 子词嵌入  
13.7. 词的相似性和类比任务  
13.8. 来自Transformers的双向编码器表示  
13.9. 用于预训练BERT的数据集  
13.10. 预训练BERT  
##### ch14.  自然语言处理：应用
14.1. 情感分析及数据集   
14.2. 情感分析：使用循环神经网络  
14.3. 情感分析：使用卷积神经网络  
14.4. 自然语言推断与数据集  
14.5. 自然语言推断：使用注意力  
14.6. 针对序列级和词元级应用程序微调BERT  
14.7. 自然语言推断：微调BERT  

