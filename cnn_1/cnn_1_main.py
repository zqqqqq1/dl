"""


简单卷积网络的直接实现
"""

import numpy as np
import matplotlib.pyplot as plt
def zero_pad(X,pad):
    """
      把数据集X的图像边界全部使用0来扩充pad个宽度和高度。

      参数：
          X - 图像数据集，维度为（样本数，图像高度，图像宽度，图像通道数）
          pad - 整数，每个图像在垂直和水平维度上的填充量
      返回：
          X_paded - 扩充后的图像数据集，维度为（样本数，图像高度 + 2*pad，图像宽度 + 2*pad，图像通道数）

      """
    X_paded = np.pad(X,(
        (0,0),#样本数 不填充
        (pad,pad),#高度 上面和下面填充pad个
        (pad,pad),#宽度，左边和右边填充pad个
        (0,0) #通道数 不填充
    ),'constant',constant_values=0) #连续值填充)

    return X_paded
# np.random.seed(1)
# X = np.random.randn(4,3,3,2)
# X_paded = zero_pad(X,2)


def conv_single_step(a_slice_prev,W,b):
    """
        在前一层的激活输出的一个片段上应用一个由参数W定义的过滤器。
        这里切片大小和过滤器大小相同

        参数：
            a_slice_prev - 输入数据的一个片段，维度为（过滤器大小，过滤器大小，上一通道数）
            W - 权重参数，包含在了一个矩阵中，维度为（过滤器大小，过滤器大小，上一通道数）
            b - 偏置参数，包含在了一个矩阵中，维度为（1,1,1）

        返回：
            Z - 在输入数据的片X上卷积滑动窗口（w，b）的结果。
        """
    s  = np.multiply(a_slice_prev,W)+b

    Z = np.sum(s)

    return Z
# np.random.seed(1)
#
# #这里切片大小和过滤器大小相同
# a_slice_prev = np.random.randn(4,4,3)
# W = np.random.randn(4,4,3)
# b = np.random.randn(1,1,1)
#
# Z = conv_single_step(a_slice_prev,W,b)
#
# print("Z = " + str(Z))

def conv_forward(A_prev, W, b, hparameters):
    """
    实现卷积的前向传播
    :param A_prev: 上一层的激活输出矩阵，维度为( m ,n_H_prev,n_W_prev,n_C,_prev)样本数量，上一层 高，宽，信道数量
    :param W: (f,f,n_C_prev,n_c) 过滤器大小 过滤器大小 上一层过滤器数量 这一层数量
    :param b: 偏置矩阵 1, 1 , 1 ,n_c
    :param hparameters: 包含了stride步伐和pad 填充的超参数字典
    :return: 
            Z:卷积输出 维度是(m , n_H, n_W,n_C)
            cache = 缓存
    
    """
    #获取上一层的基本信息
    (m , n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    #获取权重矩阵的基本信息
    (f,f,n_C_prev,n_C) = W.shape
    #获取超参数的值
    stride  = hparameters["stride"]
    pad = hparameters["pad"]

    #计算卷积后的高度，宽短
    n_H = int((n_H_prev - f + 2*pad)/stride+1)
    n_W = int((n_W_prev - f + 2 * pad) / stride + 1)

    #初始化卷积输出Z
    Z = np.zeros((m,n_H,n_W,n_C))

    #通过A_prev创建填充过了的A_prev_pad
    A_prev_pad = zero_pad(A_prev,pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h*stride
                    vert_end = vert_start+f

                    horiz_start = w*stride
                    horiz_end = horiz_start+f

                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    Z[i,h,w,c] = conv_single_step(a_slice_prev,W[:,:,:,c],b[0,0,0,c])
    assert (Z.shape ==(m,n_H,n_W,n_C))

    cache = (A_prev,W,b,hparameters)
    return (Z, cache)
# np.random.seed(1)
#
# A_prev = np.random.randn(10,4,4,3)
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(1,1,1,8)
#
# hparameters = {"pad" : 2, "stride": 1}
#
# Z , cache_conv = conv_forward(A_prev,W,b,hparameters)
#
# print("np.mean(Z) = ", np.mean(Z))
# print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
#


#池化层会减少输入的宽度和高度，这样它会较少计算量的同时也能使特征检测器对其在输入中的位置更加稳定。
#最大池化 在输入矩阵中滑动一个大小为fxf的窗口，选取窗口里的值中的最大值，然后作为输出的一部分。
#均值池划 在输入矩阵中滑动一个大小为fxf的窗口，计算窗口里的值中的平均值，然后这个均值作为输出的一部分。

def pool_forward(A_prev, hparameters,mode = "max"):
    """
    实现池化层的前向传播
    :param A_prev: 输入数据维度(m,n_H_prev,n_W_prev,n_C_prev)
    :param hparameters: 包含了f 和 s的超参数字典
    :param mode: 池划的类型 max或者average
    :return: Z: 池化层输入 维度为(m,n_H,n_W,n_C)
            cache 
    """

    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int((n_H_prev - f )/stride+1)
    n_W = int((n_W_prev - f)/stride+1)
    n_C = n_C_prev

    #初始化输出矩阵
    A = np.zeros((m,n_H,n_W,n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    #定位
                    v_start = h * stride
                    v_end = v_start + f

                    h_start = w*stride
                    h_end = h_start + f

                    a_slice_prev = A_prev[v_start:v_end,h_start:h_end,c]

                    if mode =="max":
                        A[ i,h,w,c] = np.max(a_slice_prev)
                    elif mode =="average":
                        A[i,h,w,c] = np.mean(a_slice_prev)

    #池化完毕
    assert A.shpae ==(m,n_H,n_W,n_C)

    cache = (A_prev,hparameters)
    return A,cache


#反向传播
def conv_backward(dZ,cache):
    """
    实现卷积层的反向传播
    
    :param dZ: 卷积层的输出Z 的梯度 
    :param cache: 
    :return: 
            dA_prev
            dW
            db
    """
    (A_prev,W,b,hparameters) = cache

    (m , n_H_prev , n_W_prev , n_C_prev) = A_prev.shape

    (m , n_H ,n_W , n_C) = dZ.shape

    (f ,f ,n_C_prev,n_C) = W.shape

    pad = hparameters["pad"]
    stride = hparameters["stride"]

    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    dW = np.zeros((f,f,n_C_prev,n_C))
    db = np.zeros((1,1,1,n_C))


    #前向传播我们使用了pad，反向传播也需要使用，这样是为了保持数据的一致
    A_prev_pad = zero_pad(A_prev,pad)
    dA_prev_pad = zero_pad(dA_prev,pad)

    for i in range(m):
        #选择第i个扩充了的数据的样本，降了一维
        a_prev_pad = A_prev_pad[i]
        dA_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h
                    v_end = v_start+f

                    h_start = w
                    h_end = h_start+f

                    a_slice = a_prev_pad[v_start:v_end,h_start:h_end,:]

                    #切片完毕，使用公式计算梯度

                    dA_prev_pad[v_start:v_end, h_start:h_end,:] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        dA_prev[i,:,:,:] = dA_prev_pad[pad:-pad,pad:-pad]

    assert dA_prev.shape == (m,n_H_prev,n_W_prev,n_C_prev)

    return dA_prev,dW,db

#池化层的反向传播

#处理最大池化
def create_mask_from_window(x):
    """
    从输入矩阵中创建掩码，以保存最大值的矩阵的位置
    :param x:  一个维度为 f, f 的矩阵  过滤器
    :return: 
        mask 返回最大值的位置的矩阵
    """

    mask = x == np.max(x)
    return mask

#处理平均池划
def distribute_value(dz,shape):
    """
    给定一个值，为按矩阵大小平均分配到每一个矩阵位置中
    :param dz:  输入的实数
    :param shape:  两个值分别为n_H n_w
    :return: 
    a 已经分配好了值的矩阵，里面的值全部一样
    """

    (n_H , n_W) = shape

    average = dz / (n_H * n_W)

    a = np.ones(shape) * average

    return a

def pool_backward(dA,cache,mode = "max"):
    """
    实现池化层的反向传播
    :param dA: 池化层输出的梯度， 和池化层的输出维度一样
    :param cache: 
    :param mode: 
    :return: 
        dA_prev  池化层的输入梯度，和池化层的输入梯度一样
    """

    (A_prev , hparameters ) = cache

    f = hparameters["f"]
    stride = hparameters["stride"]

    #获取A_prev 和dA_prev的基本数据信息

    (m ,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    (m, n_H,n_W,n_C) = dA.shape

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h
                    v_end = h+f

                    h_start = w
                    h_end = w+f

                    if mode =="max":
                        #切片
                        a_prev_slice = a_prev[v_start:v_end,h_start:h_end,c]

                        #掩码
                        mask = create_mask_from_window(a_prev_slice)

                        dA_prev[i,v_start:v_end,h_start,h_end,c]+=np.multiply(mask,dA[i,h,w,c])

                    elif mode =="average":
                        da = dA[i,h,w,c]

                        shape=(f,f)

                        dA_prev[i,v_start:v_end,h_start,h_end,c] +=distribute_value(dz,shape)
    assert(dA_prev.shape == A_prev.shape)

    return dA_prev
















