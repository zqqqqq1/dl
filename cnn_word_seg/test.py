# 用纯CNN实现中文分词
"""
简单步骤

Train
    1.取数据以及预处理
        读取文件。
        batch_x : 根据停止词 ，。！？；、:,\!\?;: 等将数据划分成一条条的句子 
        batch_y : 将句子进行label label是根据训练语料中已经存在的划分进行的。(S B M E)
        这样就得到了句子集合以及对应的label集合
    2.word2dic 生成一个词典
        对句子集合中的所有词进行统计，得到字频由大到小的表，并且将其保存起来
        这样就得到了一个包含了训练语料中所有字的词表
    3.CNN模型训练
        定义
        embedding_size = 128
        batch_size= 256
        epoch_num = 200
        
        计算
        3层的conv1d卷积加relu激活
        cost function 使用的是交叉熵
        optimizer使用的是Adam
        经过一定轮数的迭代训练之后，我们得到了97%的训练集正确率
        将模型参数保存起来，进行测试集上的测试
Test
    1.取数据以及预处理
        读取文件。
        batch_x : 根据停止词 ，。！？；、:,\!\?;: 等将数据划分成一条条的句子 
        batch_y : 将句子进行label label是根据训练语料中已经存在的划分进行的。(S B M E)
    2.CNN模型预测
        将得到的测试data和label 进行CNN模型的预测，得到对于所有数据中的每一个字对应的S B M E相应的概率分布
    3.维特比算法 viterbi
        维特比算法是一种动态规划算法，这里我们使用维特比算法计算最大可能的分词方式
        根据上述的到的S B M E概率可能，得到最大可能分词方式
    4.根据上述分词方式进行字词划分，并且进行输出
        
    


"""

import re
import numpy as np
import json
from collections import Counter,defaultdict
import tqdm
import os
import tensorflow as tf

def makelabel(word):
    if len(word)==1:
        return 'S'
    else :
        return 'B'+(len(word)-2)*'M'+'E'

def get_corpus(path):
    pure_tags = []
    pure_txts = []
    # 停止词
    stops = u'，。！？；、：,“”\.!\?;:\n'
    i = 0
    with open(path, 'r',encoding="UTF-8") as f:
        # line.strip(' ')  Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。

        # re.split 按停止符切分成不同的str
        txt = [line.strip(' ') for line in re.split('[' + stops + ']', f.read()) if line.strip(' ')]
        # txt中保存着由停止符切分后的不同的str
        for line in txt :
            i += 1
            if i>=50:
                return pure_txts,pure_tags
            pure_txts.append('')
            pure_tags.append('')
            for word in re.split(' +', line):
                pure_txts[-1] += word
                pure_tags[-1] += makelabel(word)

    ls = [len(i) for i in pure_txts]

    ls = np.argsort(ls)[::-1]  # 从大到小排序
    pure_txts = [pure_txts[i] for i in ls]
    pure_tags = [pure_tags[i] for i in ls]
    return pure_txts, pure_tags

p1,p2 = get_corpus('corpus_data/msr_training.utf8')
print(p1)
print(p2)