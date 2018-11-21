import numpy as np
import random
import time
from rnn_1 import cllm_utils

data = open("dinos.txt","r").read()

data = data.lower()

chars = list(set(data))

data_size ,vocab_size = len(data),len(chars)

print(chars)
print("共计有%d个字符，唯一字符有%d个"%(data_size,vocab_size))

def clip(gradients , maxValue):
    """
    使用maxValue 来修剪梯度
    :param gradients:  包含了
                        dWaa
                        dWax
                        dWya,
                        db
                        dby
    :param maxValue:  阈值 ， 把梯度限制在[-maxValue , maxValue] 中
    :return: 
            gradients 修剪后的梯度
    """
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
        'dby']


    #梯度修剪
    for gradient in [dWaa , dWax , dWya , db ,dby]:
        np.clip(gradient,-maxValue , maxValue,out=gradient)
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients

#现在假设我们的模型已经训练过了，我们希望生成新的文本

def sample(parameters ,char_to_is , seed):
    """
    根据RNN输出的概率分布序列对字符序列进行采样
    :param parameters: 
    :param char_to_is:  字符映射到索引的字典
    :param seed: 随机种子
    :return: 
    indices 包含采样字符索引的长度为n的列表
    """

    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # 步骤1
    x = np.zeros((vocab_size,1))

    #使用0初始化a_prev
    a_prev = np.zeros((n_a , 1))

    indices = []

    idx = - 1

    #循环遍历时间步骤t  在每个时间步内，从概率分布中抽取一个字符
    #并将其索引附加到 indices上，如果我们打到50个字符
    # 我们应该不可能有一个训练好的模型，我们将停止循环，这有助于调试并防止进入无限循环

    counter = 0
    newline_character = char_to_is["\n"]

    while (idx!=newline_character and counter < 50):
        #步骤2 进行前向传播
        a = np.tanh(np.dot(Wax, x )+ np.dot(Waa, a )+ b)
        z = np.dot(Wya , a )+ by
        y = cllm_utils.softmax(z)


        #设定随机种子

        np.random.seed(counter+seed)

        #步骤3 从概率分布y中抽取词汇表中字符对应的索引

        idx = np.random.choice(list(range(vocab_size)),p = y.ravel())

        #添加到索引中
        indices.append(idx)

        #步骤4 将输入字符重写为与采样索引对应的字符
        x = np.zeros((vocab_size,1))

        x[idx] = 1

        #更新a_prev 为a
        a_prev = a
        seed += 1
        counter += 1
    if counter == 50:
        indices.append(char_to_is["\n"])
    return indices















