import numpy as np
from rnn_1 import rnn_utils

def rnn_cell_forward(xt, a_prev,parameters):
    """
    RNN单元的单向前向传播
    :param xt: 时间步t 输入的数据 维度为(n_x,m）
    :param a_prev: 时间步t-1 隐藏层状态 维度为(n_a,m)
    :param parameters: 字典：包含：
                                Wax: 输入乘以权重维度为 (n_a,n_x)
                                Waa: 隐藏状态层乘以权重 维度为(n_a,n_a)
                                Wya: 隐藏状态层与输出相关的权重矩阵 (n_y,n_a)
                                ba: 偏置矩阵 (n_a,1)
                                by 输出偏置矩阵 (n_y,1)
    :return: 
                    a_next 下一个隐藏层状态 维度为 (n_a,m)
                    yt_pred 在时间步t 的预测(n_y,m)
                    cache:反向传播需要的数据元祖 （a_next,a_prev,xt,parameters)
    """
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by  = parameters["by"]

    #使用公式计算下一个激活值
    a_next = np.tanh(np.dot(Waa,a_prev) + np.dot(Wax,xt)+ ba)

    yt_pred = rnn_utils.softmax(np.dot(Wya,a_prev) + by)

    cache = (a_next , a_prev , xt , parameters)

    return a_next , yt_pred , cache

# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# Waa = np.random.randn(5,5)
# Wax = np.random.randn(5,3)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
#
# a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
# print("a_next[4] = ", a_next[4])
# print("a_next.shape = ", a_next.shape)
# print("yt_pred[1] =", yt_pred[1])
# print("yt_pred.shape = ", yt_pred.shape)


def rnn_forward(x , a0 , parameters):
    """
    
    :param x:  输入的全部数据 维度为(n_x, m ,T_x)
    :param a0: 初始化隐藏层状态 维度为(n_a, m )
    :param parameters: 
                                Wax: 输入乘以权重维度为 (n_a,n_x)
                                Waa: 隐藏状态层乘以权重 维度为(n_a,n_a)
                                Wya: 隐藏状态层与输出相关的权重矩阵 (n_y,n_a)
                                ba: 偏置矩阵 (n_a,1)
                                by 输出偏置矩阵 (n_y,1)
    :return: 
        a  所有时间步的隐藏状态 维度为(n_a,m,T_x)
        y_pred 所有时间步的预测 维度为(n_y,m ,T_x)
        caches  为反向传播的保存的元祖 维度(cache , x)
    """

    caches =  []

    #获取x和Wya的维度信息
    n_x , m , T_x = x.shape
    n_y , n_a = parameters["Wya"].shape

    #使用0来初始化a 与 y
    a = np.zeros([n_a ,m , T_x])
    y_pred = np.zeros(([n_y,m , T_x]))

    #初始化next
    a_next = a0

    #遍历所有时间步
    for t in range(T_x):

        ##1 使用rnn_cell_forward 函数来更新next隐藏状态和cache
        a_next , yt_pred , cache = rnn_cell_forward(x[:,:,t],a_next,parameters)

        #2 使用a 来保存 next 隐藏状态第 t 个位置
        a[:,:,t] = a_next

        #3 使用y来保存预测值
        y_pred[:,:,t] = yt_pred

        #4 把cache 保存到caches中
        caches.append(cache)
    caches = (caches,x)

    return a , y_pred , caches

# np.random.seed(1)
# x = np.random.randn(3,10,4)
# a0 = np.random.randn(5,10)
# Waa = np.random.randn(5,5)
# Wax = np.random.randn(5,3)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
#
# a, y_pred, caches = rnn_forward(x, a0, parameters)
# print("a[4][1] = ", a[4][1])
# print("a.shape = ", a.shape)
# print("y_pred[1][3] =", y_pred[1][3])
# print("y_pred.shape = ", y_pred.shape)
# print("caches[1][1][3] =", caches[1][1][3])
# print("len(caches) = ", len(caches))


#我们已经构建了RNN的前向传播函数，但是它还存在着梯度消失的问题


#构建一个LSTM模型

def lstm_cell_forward(xt,a_prev,c_prev,parameters):
    """
    实现lstm单元的前向传播
    :param xt: 在时间步 t 输入的数据 维度为(n_x, m )
    :param a_prev:  在时间步t-1 隐藏层状态 (n_a,m)
    :param c_prev:  在时间步t-1 的记忆状态 (n_a,m)
    :param parameters: 
            Wf 遗忘门的权值 (n_a,n_a+n_x) 
            bf 遗忘门的偏置 (n_a,1)
            Wi 更新门的权值 (n_a,n_a+n_x)
            bi 更新们的偏置 (n_a,1)
            Wc 第一个 tanh 的权值 (n_a,n_a+n_x)
            bc 第一个 tanh 的偏置 (n_a,1)
            Wo 输出门的权值 (n_a,n_a+n_x)
            bo 输出门的偏置 (n_a,1)
            Wy 隐藏状态与输出相关的权值 维度(n_y,n_a)
            by 隐藏状态与输出相关的偏置 维度(n_y,1)
    :return: 
            a_next  下一个隐藏层状态 维度为(n_a,m)
            c_next  下一个记忆状态 维度为(n_a,m)
            yt_pred 在时间步 t 的预测 维度为(n_y,m)
            cahce
    
    ft/it/ot 表示遗忘 更新 输出门 cct表示候选值 c表示记忆
    """
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # 获取 xt 与 Wy 的维度信息
    n_x, m = xt.shape
    n_y, n_a = Wy.shape


    # 1 连接a_prev与xt
    contact = np.zeros([n_a+n_x,m ])
    contact[:n_a,:] = a_prev
    contact[n_a:,:] = xt

    # 2 根据公式计算

    #遗忘门ft
    ft = rnn_utils.softmax(np.dot(Wf,contact)+ bf)

    #更新门it
    it = rnn_utils.sigmoid(np.dot(Wi,contact) + bf)

    #输出门
    ot = rnn_utils.sigmoid(np.dot(Wo,contact) + bf)

    #更新单元
    cct = np.tanh(np.dot(Wc,contact) + bc)

    #
    c_next = ft * c_prev + it * cct

    a_next = ot * np.tanh(c_next)

    #计算lstm的预测值
    yt_pred = rnn_utils.softmax(np.dot(Wy,a_next)+by)

    cache = (a_next , c_next , a_prev ,c_prev, ft , it , cct ,ot ,xt , parameters)

    return a_next , c_next , yt_pred ,cache

# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# c_prev = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
# print("a_next[4] = ", a_next[4])
# print("a_next.shape = ", c_next.shape)
# print("c_next[2] = ", c_next[2])
# print("c_next.shape = ", c_next.shape)
# print("yt[1] =", yt[1])
# print("yt.shape = ", yt.shape)
# print("cache[1][3] =", cache[1][3])
# print("len(cache) = ", len(cache))
#
#


def lstm_forward(x , a0, parameters):
    """
    实现LSTM 单元组成的RNN循环神经网络
    :param x:  所有时间步的输入数据 维度为(n_x , m , T_x)
    :param a0: 初始化隐藏状态 维度(n_a, m )
    :param parameters: 
             Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                        bf -- 遗忘门的偏置，维度为(n_a, 1)
                        Wi -- 更新门的权值，维度为(n_a, n_a + n_x)
                        bi -- 更新门的偏置，维度为(n_a, 1)
                        Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                        bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                        Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                        bo -- 输出门的偏置，维度为(n_a, 1)
                        Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                        by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)
    :return: 
                        a  所有时间步的隐藏状态 维度为(n_a , m, T_x)
                        y  所有时间步的预测值   维度为(n_y , m ,T_x）
                        caches 
    """
    caches = []
    n_x , m ,T_x = x.shape
    n_y , n_a = parameters["Wy"].shape

    a = np.zeros([n_a , m, T_x ])
    c = np.zeros([n_a , m, T_x ])
    y = np.zeros([n_y , m, T_x ])

    a_next = a0
    c_next = np.zeros([n_a , m])


    for t in range(T_x):
        #钢芯下一个隐藏层状态
        a_next ,c_next , yt_pred, cache = lstm_cell_forward(x[:,:,t], a_next, c_next , parameters)

        a[:,:,t] = a_next

        c[:,:,t] = c_next

        y[:,:,t] = yt_pred

        caches.append(cache)

    caches = (caches , x)
    return a, y,c , caches
# np.random.seed(1)
# x = np.random.randn(3,10,7)
# a0 = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a, y, c, caches = lstm_forward(x, a0, parameters)
# print("a[4][3][6] = ", a[4][3][6])
# print("a.shape = ", a.shape)
# print("y[1][4][3] =", y[1][4][3])
# print("y.shape = ", y.shape)
# print("caches[1][1[1]] =", caches[1][1][1])
# print("c[1][2][1]", c[1][2][1])
# print("len(caches) = ", len(caches))
#



#RNN 的反向传播
def rnn_cell_backward(da_next , cache):
    """
    实现Rnn单元的反向传播
    :param da_next:  下一个隐藏状态的损失的梯度
    :param cache: 
    :return: 
            
            dx   输入数据的梯度，维度为(n_x, m )
            dx -- 输入数据的梯度，维度为(n_x, m)
                        da_prev -- 上一隐藏层的隐藏状态，维度为(n_a, m)
                        dWax -- 输入到隐藏状态的权重的梯度，维度为(n_a, n_x)
                        dWaa -- 隐藏状态到隐藏状态的权重的梯度，维度为(n_a, n_a)
                        dba -- 偏置向量的梯度，维度为(n_a, 1)
    """

    a_next , a_prev , xt , parameters = cache

    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    dtanh = (1 - np.square(a_next)) * da_next

    # 计算关于Wax损失的梯度
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    # 计算关于Waa损失的梯度
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    # 计算关于b损失的梯度
    dba = np.sum(dtanh, keepdims=True, axis=-1)

    # 保存这些梯度到字典内
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients


def rnn_backward(da, caches):
    """
    在整个输入数据序列上实现RNN的反向传播

    参数：
        da -- 所有隐藏状态的梯度，维度为(n_a, m, T_x)
        caches -- 包含向前传播的信息的元组

    返回：    
        gradients -- 包含了梯度的字典：
                        dx -- 关于输入数据的梯度，维度为(n_x, m, T_x)
                        da0 -- 关于初始化隐藏状态的梯度，维度为(n_a, m)
                        dWax -- 关于输入权重的梯度，维度为(n_a, n_x)
                        dWaa -- 关于隐藏状态的权值的梯度，维度为(n_a, n_a)
                        dba -- 关于偏置的梯度，维度为(n_a, 1)
    """
    # 从caches中获取第一个cache（t=1）的值
    caches, x = caches
    a1, a0, x1, parameters = caches[0]

    # 获取da与x1的维度信息
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # 初始化梯度
    dx = np.zeros([n_x, m, T_x])
    dWax = np.zeros([n_a, n_x])
    dWaa = np.zeros([n_a, n_a])
    dba = np.zeros([n_a, 1])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])

    # 处理所有时间步
    for t in reversed(range(T_x)):
        # 计算时间步“t”时的梯度
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])

        # 从梯度中获取导数
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients[
            "dWaa"], gradients["dba"]

        # 通过在时间步t添加它们的导数来增加关于全局导数的参数
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    # 将 da0设置为a的梯度，该梯度已通过所有时间步骤进行反向传播
    da0 = da_prevt

    # 保存这些梯度到字典内
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients


def lstm_cell_backward(da_next, dc_next, cache):
    """
    实现LSTM的单步反向传播

    参数：
        da_next -- 下一个隐藏状态的梯度，维度为(n_a, m)
        dc_next -- 下一个单元状态的梯度，维度为(n_a, m)
        cache -- 来自前向传播的一些参数

    返回：
        gradients -- 包含了梯度信息的字典：
                        dxt -- 输入数据的梯度，维度为(n_x, m)
                        da_prev -- 先前的隐藏状态的梯度，维度为(n_a, m)
                        dc_prev -- 前的记忆状态的梯度，维度为(n_a, m, T_x)
                        dWf -- 遗忘门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbf -- 遗忘门的偏置的梯度，维度为(n_a, 1)
                        dWi -- 更新门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbi -- 更新门的偏置的梯度，维度为(n_a, 1)
                        dWc -- 第一个“tanh”的权值的梯度，维度为(n_a, n_a + n_x)
                        dbc -- 第一个“tanh”的偏置的梯度，维度为(n_a, n_a + n_x)
                        dWo -- 输出门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbo -- 输出门的偏置的梯度，维度为(n_a, 1)
    """
    # 从cache中获取信息
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    # 获取xt与a_next的维度信息
    n_x, m = xt.shape
    n_a, m = a_next.shape

    # 根据公式7-10来计算门的导数
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

    # 根据公式11-14计算参数的导数
    concat = np.concatenate((a_prev, xt), axis=0).T
    dWf = np.dot(dft, concat)
    dWi = np.dot(dit, concat)
    dWc = np.dot(dcct, concat)
    dWo = np.dot(dot, concat)
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)

    # 使用公式15-17计算洗起来了隐藏状态、先前记忆状态、输入的导数。
    da_prev = np.dot(parameters["Wf"][:, :n_a].T, dft) + np.dot(parameters["Wc"][:, :n_a].T, dcct) + np.dot(
        parameters["Wi"][:, :n_a].T, dit) + np.dot(parameters["Wo"][:, :n_a].T, dot)

    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next

    dxt = np.dot(parameters["Wf"][:, n_a:].T, dft) + np.dot(parameters["Wc"][:, n_a:].T, dcct) + np.dot(
        parameters["Wi"][:, n_a:].T, dit) + np.dot(parameters["Wo"][:, n_a:].T, dot)

    # 保存梯度信息到字典
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients


def lstm_backward(da, caches):
    """
    实现LSTM网络的反向传播

    参数：
        da -- 关于隐藏状态的梯度，维度为(n_a, m, T_x)
        cachses -- 前向传播保存的信息

    返回：
        gradients -- 包含了梯度信息的字典：
                        dx -- 输入数据的梯度，维度为(n_x, m，T_x)
                        da0 -- 先前的隐藏状态的梯度，维度为(n_a, m)
                        dWf -- 遗忘门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbf -- 遗忘门的偏置的梯度，维度为(n_a, 1)
                        dWi -- 更新门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbi -- 更新门的偏置的梯度，维度为(n_a, 1)
                        dWc -- 第一个“tanh”的权值的梯度，维度为(n_a, n_a + n_x)
                        dbc -- 第一个“tanh”的偏置的梯度，维度为(n_a, n_a + n_x)
                        dWo -- 输出门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbo -- 输出门的偏置的梯度，维度为(n_a, 1)

    """

    # 从caches中获取第一个cache（t=1）的值
    caches, x = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    # 获取da与x1的维度信息
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # 初始化梯度
    dx = np.zeros([n_x, m, T_x])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])
    dc_prevt = np.zeros([n_a, m])
    dWf = np.zeros([n_a, n_a + n_x])
    dWi = np.zeros([n_a, n_a + n_x])
    dWc = np.zeros([n_a, n_a + n_x])
    dWo = np.zeros([n_a, n_a + n_x])
    dbf = np.zeros([n_a, 1])
    dbi = np.zeros([n_a, 1])
    dbc = np.zeros([n_a, 1])
    dbo = np.zeros([n_a, 1])

    # 处理所有时间步
    for t in reversed(range(T_x)):
        # 使用lstm_cell_backward函数计算所有梯度
        gradients = lstm_cell_backward(da[:, :, t], dc_prevt, caches[t])
        # 保存相关参数
        dx[:, :, t] = gradients['dxt']
        dWf = dWf + gradients['dWf']
        dWi = dWi + gradients['dWi']
        dWc = dWc + gradients['dWc']
        dWo = dWo + gradients['dWo']
        dbf = dbf + gradients['dbf']
        dbi = dbi + gradients['dbi']
        dbc = dbc + gradients['dbc']
        dbo = dbo + gradients['dbo']
    # 将第一个激活的梯度设置为反向传播的梯度da_prev。
    da0 = gradients['da_prev']

    # 保存所有梯度到字典变量内
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients

