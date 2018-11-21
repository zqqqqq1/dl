import numpy as np
from keras import layers
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalAvgPool2D,GlobalMaxPool2D
from keras.models import Model
from keras.preprocessing import  image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from cnn_2_keras import kt_utils

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = kt_utils.load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

def model(input_shape):
    """
    模型大纲
    :param input_shape: 
    :return: 
    """
    #定义一个tensor的placeholder 维度为input_shape
    X_input = Input(input_shape)

    #使用0填充，X_input周围填充0
    X =ZeroPadding2D((3,3,))(X_input)

    #对X使用Conv -》 BN -》RELU块

    X = Conv2D(32,(7,7),strides=(1,1),name = 'conv0')(X)
    X = BatchNormalization(axis=3, name = 'bn0')(X)
    X = Activation('relu')(X)

    #最大值池化层
    X = MaxPooling2D((2,2) , name = "max_pool")(X)

    #降维
    X = Flatten()(X)
    X = Dense(1,activation="sigmoid",name="fc")(X)

    model = Model(inputs = X_input , outputs = X,name="HappyModel")

    return model

def HappyModel(input_shape):
    """
    实现一个检测笑容的模型
    :param input_shape: 
    :return: 
    """
    X_input = Input(input_shape)

    #使用0 填充
    X = ZeroPadding2D((3,3))(X_input)

    #对X使用CONV -> BN ->RELU
    X = Conv2D(32,(7,7),strides=(1,1),name="conv0")(X)
    X = BatchNormalization(axis=3,name="bn0")(X)
    X = Activation('relu')(X)

    #最大值池化层
    X = MaxPooling2D((2, 2),name='max_pool')(X)

    #降维：
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid' , name= 'fc')(X)

    #创建模型
    model = Model(inputs = X_input , outputs = X, name='HappyModel')
    return model

happy_model = HappyModel(X_train.shape[1:])

happy_model.compile("adam","binary_crossentropy",metrics= ['accuracy'])

happy_model.fit(X_train,Y_train,epochs=40,batch_size=50)
preds = happy_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
print ("误差值 = " + str(preds[0]))
print ("准确度 = " + str(preds[1]))




x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happy_model.predict(x))



































