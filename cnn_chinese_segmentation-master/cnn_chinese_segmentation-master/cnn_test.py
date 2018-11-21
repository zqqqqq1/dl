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


import numpy as np
import tensorflow as tf
from cnn_word_seg import cnn_train
import json

#读取数据
def get_test_data(pure_txts,pure_tags,word_id,tag2vec):
    x=[]
    y=[]
    for i in range(len(pure_txts)):
        #extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
        x.extend([word_id.index(j,4726) for j in pure_txts[i]])
        y.extend([tag2vec[j] for j in pure_tags[i]])
    return [x],[y]


#cnn模型
def model_test(vac_size,x_data,y_data,predict=False):
    embedding_size=128
    keep_prob=tf.placeholder(tf.float32)
    embeddings=tf.Variable(tf.random_uniform([vac_size,embedding_size],-1,1))

    x=tf.placeholder(tf.int32,shape=[None,None])
    embedded=tf.nn.embedding_lookup(embeddings,x)
    embedded_dropout=tf.nn.dropout(embedded,keep_prob)

    W1=tf.Variable(tf.random_uniform([3,embedding_size,embedding_size],-1,1))
    b1=tf.Variable(tf.random_uniform([embedding_size],-1,1))
    a1=tf.nn.relu(tf.nn.conv1d(embedded_dropout,W1,stride=1,padding='SAME')+b1)

    W2=tf.Variable(tf.random_uniform([3,embedding_size,int(embedding_size/4)],-1,1))
    b2=tf.Variable(tf.random_uniform([int(embedding_size/4)],-1,1))
    a2=tf.nn.relu(tf.nn.conv1d(a1,W2,stride=1,padding='SAME')+b2)

    W3=tf.Variable(tf.random_uniform([3,int(embedding_size/4),4],-1,1))
    b3=tf.Variable(tf.random_uniform([4],-1,1))
    y_pre=tf.nn.softmax(tf.nn.conv1d(a2,W3,stride=1,padding='SAME')+b3)

    y=tf.placeholder(tf.float32,shape=[None,None,4])

    sess=tf.Session()
    saver=tf.train.Saver()
    #读取训练好的模型
    saver.restore(sess,'./model_data/model_data/frist_model.ckpt')

    # print(sess.run(W3),W3.shape)
    correct_pre=tf.equal(tf.argmax(y,2),tf.argmax(y_pre,2))
    acc=tf.reduce_mean(tf.cast(correct_pre,tf.float32))

    if predict:
        result = sess.run(y_pre, feed_dict={x: x_data, keep_prob: 0.5})
        #得到的句子中每一个词对应的S B M E的概率
        #print(result)
        return result

    else:
        sess.run(y_pre, feed_dict={x: x_data, keep_prob: 0.5})
        scores = sess.run(acc, feed_dict={x: x_data, y: y_data, keep_prob: 1.0})
        print("Test_data accuracy: " , scores)



#维特比算法，动态规划算法，找到分词的可能性最大的分词方法
def viterbi(result,trans_pro):
    #将句子结果词和S B M E连接起来
    nodes=[dict(zip( ('S','B','M','E'),i )) for i in result]
    #print(nodes)

    #路径的开始
    paths=nodes[0]
    #循环寻找 从1开始 （第二个）
    for t in range(1,len(nodes)):
        #print("当前 " ,t)

        #path_old 开始到当前点所有分词类型 和可能
        path_old=paths.copy()
        #print(path_old)
        paths={}


        #print("nodes[t]")
        #print(nodes[t])
        #nodes[t] 表示当前词被分为 SBEM 的四种可能性


        #在当前词的4中可能中遍历
        for i in nodes[t]:
            nows={}

            #从以往的点中
            for j in path_old:
               # print(trans_pro)
                #如果i-》j 是可以到达的 在BESM中存在可能
                if j[-1]+i in trans_pro:
                    nows[j+i]=path_old[j]+nodes[t][i]+trans_pro[j[-1]+i]
            #找到对于当前点 到达的点中可能性最大的点
            pro,key=max([(nows[key],key) for key,value in nows.items()])

            paths[key]=pro
    best_pro,best_path=max([(paths[key],key)for key,value in paths.items()])
    #print(best_path)
    #print(best_pro)
    #best_path 最佳路径
    return best_path

def segword(txt,best_path):
    begin,end=0,0
    seg_word=[]
    for index,char in enumerate(txt):
        signal=best_path[index]

        #当前为B 词的开始
        if signal=='B':
            begin=index
        #当前为E 词的结尾
        elif signal=='E':
            #添加上次begin到当前的所有字 为一个词
            seg_word.append(txt[begin:index+1])
            end=index+1
        #当前为S solo词 单字词
        elif signal=='S':
            #直接添加
            seg_word.append(char)
            end=index+1
    #还有未进行添加的词 直接添加
    if end<len(txt):
        seg_word.append(txt[end:])
    return seg_word

def cnn_seg(txt):
    word_id=json.load(open('vacabulary.json','r'))
    print(type(word_id))
    vacabulary_size=len(word_id)+1
    trans_pro={'SS':1,'BM':1,'BE':1,'SB':1,'MM':1,'ME':1,'EB':1,'ES':1}
    trans_pro={state:np.log(num) for state,num in trans_pro.items()}
    #print(trans_pro)
    txt2id=[[word_id.get(word,4726)for word in txt]]
    print(type(txt2id))
    print(txt2id)
    result=model_test(vacabulary_size,x_data=txt2id,y_data=None,predict=True)
    result = result[0, :, :]
    best_path=viterbi(result,trans_pro)

    return  segword(txt,best_path)

def cnn_test(path="./corpus_data/msr_test_gold.utf8"):
    #获取训练数据
    pure_txts, pure_tags = cnn_train.get_corpus(path)
    #获取词序号id
    word_id=json.load(open('vacabulary.json','r',encoding="UTF-8"))
    #获取词表大小
    vacabulary_size=len(word_id)+1
    #tag2vec
    tag2vec = {'S': [1, 0, 0, 0], 'B': [0, 1, 0, 0], 'M': [0, 0, 1, 0], 'E': [0, 0, 0, 1]}
    x, y = get_test_data(pure_txts, pure_tags, word_id, tag2vec)

    model_test(vacabulary_size, x_data=x, y_data=y)

if __name__ == '__main__':
    #print(cnn_test())

    ##测试句子分词效果
    print(cnn_seg("我爱自然语言处理"))

    ## 模型在测试集上的效果
    #print(cnn_test())
    
    #-----以下为各种函数的编写过程---------------
    #获取测试集的数据
    # pure_txts,pure_tags=cnn_crf.get_corpus("./corpus_data/msr_test_gold.utf8")
    # tf_config=cnn_crf.make_default()
    # word_id=json.load(open('vacabulary.json','r'))
    # vacabulary_size=len(word_id)+1
    # print(vacabulary_size)
    # tag2vec = {'S': [1, 0, 0, 0], 'B': [0, 1, 0, 0], 'M': [0, 0, 1, 0], 'E': [0, 0, 0, 1]}
    # x, y = get_test_data(pure_txts, pure_tags, word_id, tag2vec)
    # # model_test(vacabulary_size,x_data=x,y_data=y)
    #
    # trans_pro={'SS':1,'BM':1,'BE':1,'SB':1,'MM':1,'ME':1,'EB':1,'ES':1}
    # trans_pro={state:np.log(num) for state,num in trans_pro.items()}
    # print(trans_pro)

    # txt="我爱中国"
    # test_txt="我爱中国"
    # test_txt=[[word_id.get(word,4726)for word in test_txt]]
    # result=model_test(vacabulary_size,x_data=test_txt,y_data=None,predict=True)
    # result=result[0,:,:]

    # nodes=np.random.random((6,4))
    # print(nodes)
    # nodes=[dict(zip(('S','B','M','E') , i)) for i in result]
    # paths=nodes[0]
    # # print(trans_pro['ss'])
    # for t in range(1,len(nodes)):
    #     path_old=paths.copy()
    #     paths={}
    #     for i in nodes[t]:
    #         nows={}
    #         for j in path_old:
    #             if j[-1]+i in trans_pro:
    #                 nows[j+i]=path_old[j]+nodes[t][i]+trans_pro[j[-1]+i]
    #         pro,key=max([(nows[key],key) for key,value in nows.items()])
    #         # print(nows,pro,key)
    #         paths[key]=pro
    # best_pro,best_path=max([(paths[key],key) for key,value in paths.items()])
    # print(best_path)
    # best_path=viterbi(result)
    # print(segword(txt,best_path))

    # seg_word=[]
    # start,end=0,0
    # for index,char in enumerate(txt):
    #     signal=best_path[index]
    #     if signal=='B':
    #         begin=index
    #     elif signal=='E':
    #         seg_word.append(txt[begin:index+1])
    #         end=index+1
    #     elif signal=='S':
    #         seg_word.append(char)
    #         end=index+1
    # if end<len(txt):
    #     seg_word.append(txt[end:])
    # print(seg_word)



    # Viterbi version-1
    #S-0,B-1,M-2,E-3
    #第一个节点初始化
    # paths=nodes[0]
    # print(paths)

    # test_txt="虽然一路上队伍里肃静无声"
    # test_txt=[[word_id.get(word,4726)for word in test_txt]]
    # result=model_test(vacabulary_size,x_data=test_txt,y_data=None,predict=True)
    # result=result[0,:,:]
    # print(result.shape,'\n',result)
    # nodes=result
    # path_v=[{}]
    # path_max={}
    # for state,i in zip(states,range(len(states))):
    #     path_v[0][state]=nodes[0][i]
    #     path_max[state]=[state]
    # for t in range(1,len(nodes)):
    #     path_v.append({})
    #     new_path={}
    #     for state in states:
    #         (temp_pro,temp_state)=max([(path_v[t-1][y]+nodes[t][k],y) for y,k in zip(states,range(len(states)))])
    #         path_v[t][state]=temp_pro
    #         new_path[state]=path_max[temp_state]+[state]
    #     path_max=new_path
    # best_path_pro,last_state=max([(path_v[len(nodes)-1][y0],y0) for y0,j in zip(states,range(len(states)))])
    # print(path_max[last_state])
