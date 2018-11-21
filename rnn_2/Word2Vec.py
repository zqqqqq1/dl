#下面开始用TensorFlow实现Word2Vec的训练
#首先依然是载入各种依赖库
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf
import matplotlib.pyplot as plt
#定义下载文本数据的函数，这里使用urllib.request.urlretrieve
#url = 'http://mattmahoney.net/dc/'

# def maybe_download(filename,expected_bytes):
#     if not os.path.exists(filename):
#         filename , _ = urllib.request.urlretrieve(url + filename,filename)
#     statinfo = os.stat(filename)
#     if statinfo.st_size == expected_bytes:
#         print('Found and verified' , filename)
#     else:
#         print(statinfo.st_size)
#         raise Exception(
#             'Failed to verify' + filename +'. Can you get to it with a browser?'
#         )
#     return filename
# filename = maybe_download('text8.zip',31344016)
#接下来解压下载的压缩问价你，并且使用tf.compat.as_st将数据转成单词的李彪。
#通过程序输出，可以知道数据最后被转为了一个包含17005207个单词的列表

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
# with open('text8.txt') as f:
#     data = f.read()
#     words = tf.compat.as_str(f.namelist()[0]).split()
filename = 'text8.zip'
words = read_data(filename)

print('Data Size',len(words))

#接下来创建vocabulary词汇表，我们使用collections.Counter统计单词列表中的单词
#的频数，然后使用most_common方法取top50000频数的单词作为vocabulary
#再创建一个dict，将top50000词汇放入dictionary中，便于快速查询
#接下来将全部单词转为编号（频数）
#top50000之外的单词，我们认定其为Unkown，将其编号为0
#并且统计这类词汇的数量
#下面遍历单词列表，对其中每一个单词，先判断是否出现在dictionary中，
#如果是则转为其编号，如果不是则转为0,
#最后返回转换后的编码data，每个单词的频数统计count，词汇表dictionary
#及其反转之后的形式reverse_dictionary

vocabulary_size = 50000

def build_dataset(words):
    count  = [['UNK', -1]]
    #取前50000个出现最频繁的词
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data, count , dictionary,reverse_dictionary
data , count ,dictionary ,reverse_dictionary = build_dataset(words)
#然后我们删除原始单词列表，可以节约内存
#打印出vocabulary中最高频出现的单词和其数量(包括Unkown）
del words
print('most common words （+unk)',count[:5])
print('sample data',data[:10],[reverse_dictionary[i] for i in data[:10]])

#下面生成Word2Vec的训练样本。我们根据前面提到的Skip-Gram模式（从目标单词反推语境）
#将原始数据"the quick brwon fox for jumped over the lazy dog"转换成(quick ,the))
#(quick ,brwon)等样本
#我们定义函数generate_batch 用来升成训练用的batch数据
#参数中batch_size为batch的大小；
#skip_window指单词最远可以联系的距离，设为1代表只能跟紧邻的两个单词生成样本
#比如quick只能和前后的单词生成
#num_skips 为对每个单词生成多少个样本
#它不能够大于skip_window值得两倍
#并且batch_size必须是它的整数倍
#我们定义单词序号data_index 为global变量因为我们会反复调用generate_batch
#所以要确保data_index可以在函数generate_batch中被修改
#我们也会使用assert确保num_skips和batch_size满足前面的条件
#然后使用np》ndarray将batch和labels初始化为数据
#这里定义span为对某个单词出那个键相关样本时会使用到的单词数量
#包括目标单词本身和它前后的单词，因此span = 2 * skip_window +1
#并且创建一个最大容量为span的deque，即双向队列
#在对deque使用append方法添加变量时，只会保留最后插入的span个变量

data_index = 0
def generate_batch(batch_size , num_skips , skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span = 2 * skip_window +1
    buffer = collections.deque(maxlen=span)
#接下来从序号data_index开始，把span个单词顺序读入buffer作为初始值
#因为buffer是容量为span的deque，所以此时buffer已经填充满了
#后续数据将替换掉前面的数据
#然后我们进入第一层循环（次数为batch_size//num_skips）
#每次循环内对一个目标单词生成样本
#现在buffer中是目标单词和所有相关单词，我们定义target = skip_window
#即wbuffer中的地skip_window个变量为目标单词
#然后定义生成样本时需要避免的单词列表targets_to_avoid
#这个列表一开始包括第skip_window个单词，即目标单词
#因为我们要预测的是语境单词，不包括目标单词本身。
#接下里进入第二层循环(次数为num_skips)
#每次循环红对一个语境单词生成样本，先产生随机数，会自动啊随机数不在tartgets_to_avoid中
#代表可以使用的语境单词，然后产生一个样本
#feature 即目标词汇buffer[skip_window],
#label 则是buffer[target]
#同时因为这个语境词已经被使用了，所以再把它添加到targets_to_avoid中过滤
#在对一个怒表单词生成完所有样本之后（num_skips个样本）
#我们再读入下一个单词（同时会跑调buffer中的第一个单词——，即把滑窗向后移动一位
#这样我们的目标单词也想后移动了一位，我们已经获得了batch_size个训练样本
#将batch和labels作为结果返回。


    #span是窗口加上目标单词 如果窗口是±2，则span  = 2*2 +1;
    for _ in range(span):
        #将目标单词和左右窗口单词放入
        buffer.append(data[data_index])
        data_index = (data_index + 1 )%len(data)
    # //取整除

    #对一个目标单词生成样本
    #现在buffer中是目标单词和所有的相关单词
    for i in range(batch_size // num_skips):
        #因为当前span包括了左窗口，目标单词，右窗口
        #所以skip_window （窗口大小）为序号的就是目标单词

        target = skip_window
        #定义一个数组，包含目标单词和已经处理的窗口单词，防止重复操作
        targets_to_avoid = [skip_window]

        #第二层循环
        #每一次循环对一个语境单词生成样本，产生随机数，直到随机数不在已经操作过的数据中
        #然后产生一个样本，feature 即目标词汇buffer[skip_window],label则是buffer[target]
        #一个单词会生成nums_skips个样本
        #仅仅只统计了目标单词周围 窗口之内的单词，并没有进行其他 统计操作
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0,span-1)
            targets_to_avoid.append(target)
            batch[i * num_skips+j] = buffer[skip_window]
            labels[i * num_skips+j,0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1)% len(data)

        #对一个目标词汇完成操作之后，我们窗口后滑一位，目标词后移一位，语境单词也整体后移了
    return batch , labels
#这里调用generate_bacth 函数简单测试一下其功能
#参数中将batch_size设为8，num_skip设为2，skip_window 1
#获取测试结果


#@batch_size 样本大小
#num_skips 每个目标词产生的样本
#左右窗口大小
batch , labels  =generate_batch(batch_size=8,num_skips=2,skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],' - > ',labels[i,0],
          reverse_dictionary[labels[i,0]])

#目标词样本大小
batch_size = 128
#单词转为稠密向量的维度
embedding_size = 128
#单词最远可以联系到的距离 即窗口大小
skip_window = 1
#每个目标单词提取的样本数
num_skips = 2
#生成验证数据 valid_examples
#随机抽取一些频数最高的单词，看向量空间上跟它们最近的单词是否相关性比较高

#抽取的验证单词数
valid_size = 16
#验证单词只从频数最高的100个单词中抽取
valid_window = 100
#进行随机抽取
valid_examples = np.random.choice(valid_window,valid_size,replace=False)
#训练时用来做负样本的噪声单词的数量
num_sampled = 64



#下面开始定义skip_gram word2vec模型的网络结构

#我们先创建一个tf.Graph并设置为默认的graph
with tf.device('/cpu:0'):
    graph = tf.Graph()
    with graph.as_default():
        # 创建placeholder
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        # 将前面随机产生的valid_examples转为tensorflow中的constant
        valid_dateset = tf.constant(valid_examples, dtype=tf.int32)

        # 限定计算操作在cpu上执行，因为接下来的操作可能在gpu上还没有实现
        # 使用tf.random_uniform随机生成所有单词的词向量embeddings
        # 单词表大小为50000，向量维度为128


        # 生成一个 n*m的矩阵 每个值大小介于 -1  1
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        # 再使用tf.nn.embedding_loopup 查找输入train_inputs 对应的向量embed
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # 下面使用之前提到了NCE Loss作为训练的优化目标，我们使用tf.truncated_normal初始化NCE loss计算学习出的词向量embeeding
        # 在训练数据上的loss，并且使用tf.reduce_mean进行汇总
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size))
        )

        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=train_labels,
                                             inputs=embed,
                                             num_sampled=num_sampled,
                                             num_classes=vocabulary_size))

        # 我们定义优化器为SGD,并且学习速率为1.0
        optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

        # 然后计算嵌入向量embeddings的L2范式norm
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

        # 再将embeddings 除以其L2范数 得到标准化后的normalized_embeddings
        normalized_embeddings = embeddings / norm
        # 再使用tf.nn.enbedding_lookup  查询验证单词的嵌入向量
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dateset
        )
        # 计算验证单词的嵌入向量与词汇表中所有单词的相似性
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True
        )

        init = tf.global_variables_initializer()
        # 我们定义最大迭代的次数为10万次
        # 然后创建并且声称默认的session，并且执行参数的初始化
        # 在每一步的训练迭代中，先使用generate_batch生成一个batch的inputs和labels数据
        # 并用它们创建feed_dict，然后使用session.run()执行一个优化器运算(即一次参数更新)和损失运算
        # 并将这一步训练的loss累加到average_loss

        num_steps = 100001
        with tf.Session(graph=graph)as session:
            init.run()
            print("Initialized")

            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = generate_batch(
                    batch_size, num_skips, skip_window
                )
                feed_dict = {train_inputs: batch_inputs,
                             train_labels: batch_labels}

                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                # 之后每2000次循环，计算一下平均loss并且显示
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                        print("Average loss as sttep ", step, ": ", average_loss)
                        average_loss = 0

                # 每10000次循环，计算一次验证单词与全部单词的相似度，并将与每个验证单词最相似的8个单词展示出来
                if step % 10000 == 0:
                    if step > 0:
                        sim = similarity.eval()
                        for i in range(valid_size):
                            valid_word = reverse_dictionary[valid_examples[i]]
                            top_k = 8
                            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                            log_str = "Nearest to %s:" % valid_word

                            for k in range(top_k):
                                close_word = reverse_dictionary[nearest[k]]
                                log_str = "%s,%s," % (log_str, close_word)
                                # print(log_str)

            final_embeddings = normalized_embeddings.eval()
            # 下面定义一个用来可视化Word2vec效果的函数
            # 这里的low_dim_embs 是降维到2维的单词的空间向量
            # 我们将在图标中展示每个单词的位置，我们使用plt.scatter显示散点图（单词的位置）


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
                label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom'
            )
    plt.savefig(filename)


    # 我们使用sklean.manifold.TSNE 实现降维
    # 这里直接将原始的128维嵌入向量降到2维，再用上述的函数进行表示
    # 这里只展示词频最高的100个单词的可视化结果
from sklearn.manifold import TSNE
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)





