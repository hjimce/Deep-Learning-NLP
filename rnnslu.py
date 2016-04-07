#coding=utf-8
from collections import OrderedDict
import cPickle
import os
import random
import numpy
import theano
from theano import tensor as T
from Preprocess.LoadData import  Segment



# 打乱样本数据
def shuffle(lol, seed):
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


#输入一个长句，我们根据窗口获取每个win内的数据，作为一个样本。或者也可以称之为作为RNN的某一时刻的输入
def contextwin(l, win):
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]#在一个句子的末尾、开头，可能win size内不知，我们用-1 padding
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out




# 输出结果，用于脚本conlleval.pl的精度测试，该脚本需要自己下载，在windows下调用命令为:perl conlleval.pl < filename
def conlleval(p, g, w, filename):
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()




class RNNSLU(object):
    def __init__(self, nh, nc, ne, de, cs):
        '''
        nh ::隐藏层神经元个数
        nc ::输出层标签分类类别
        ne :: 单词的个数
        de :: 词向量的维度
        cs :: 上下文窗口
        '''
        #词向量实际为(ne, de)，外加1行，是为了边界标签-1而设定的
        self.emb = theano.shared(name='embeddings',value=0.2 * numpy.random.uniform(-1.0, 1.0,(ne+1, de)).astype(theano.config.floatX))#词向量空间
        self.wx = theano.shared(name='wx',value=0.2 * numpy.random.uniform(-1.0, 1.0,(de * cs, nh)).astype(theano.config.floatX))#输入数据到隐藏层的权重矩阵
        self.wh = theano.shared(name='wh', value=0.2 * numpy.random.uniform(-1.0, 1.0,(nh, nh)).astype(theano.config.floatX))#上一时刻隐藏到本时刻隐藏层循环递归的权值矩阵
        self.w = theano.shared(name='w',value=0.2 * numpy.random.uniform(-1.0, 1.0,(nh, nc)).astype(theano.config.floatX))#隐藏层到输出层的权值矩阵
        self.bh = theano.shared(name='bh', value=numpy.zeros(nh,dtype=theano.config.floatX))#隐藏层偏置参数
        self.b = theano.shared(name='b',value=numpy.zeros(nc,dtype=theano.config.floatX))#输出层偏置参数

        self.h0 = theano.shared(name='h0',value=numpy.zeros(nh,dtype=theano.config.floatX))


        self.params = [self.emb, self.wx, self.wh, self.w,self.bh, self.b, self.h0]#所有待学习的参数

        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y_sentence = T.ivector('y_sentence')  # 训练样本标签

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)#通过ht-1、x计算隐藏层
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)#计算输出层
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])
        #神经网络的输出
        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)


        #计算损失函数，然后进行梯度下降
        lr = T.scalar('lr')#学习率，一会儿作为输入参数
        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)[T.arange(x.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g) for p, g in zip(self.params, sentence_gradients))


        #构造预测函数、训练函数，输入数据idxs每一行是一个样本(也就是一个窗口内的序列索引)
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],outputs=sentence_nll,updates=sentence_updates)
        #词向量归一化，因为我们希望训练出来的向量是一个归一化向量
        self.normalize = theano.function(inputs=[],updates={self.emb:self.emb /T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0, 'x')})

    #训练
    def train(self, x, y, window_size, learning_rate):

        cwords = contextwin(x, window_size)#获取训练样本
        words = map(lambda x: numpy.asarray(x).astype('int32'), cwords)#数据格式转换
        labels = y

        self.sentence_train(words, labels, learning_rate)
        self.normalize()

    #保存、加载训练模型
    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())
    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))


def main():

    train_set, valid_set,test_set, dicts = cPickle.load(open("atis.fold3.pkl//atis.fold3.pkl"))#加载训练数据
    idx2label = dict((k, v) for v, k in dicts['labels2idx'].iteritems())#每个类别标签名字所对应的索引
    idx2word = dict((k, v) for v, k in dicts['words2idx'].iteritems())#每个词的索引
    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex, test_ne, test_y = test_set

    #计算相关参数
    vocsize = len(set(reduce(lambda x, y: list(x) + list(y),train_lex + valid_lex + test_lex)))#计算词的个数
    nclasses = len(set(reduce(lambda x, y: list(x)+list(y),train_y + test_y + valid_y)))#计算样本类别个数
    winsize=7#窗口大小
    ndim=50#词向量维度
    nhidden=200#隐藏层的神经元个数
    learn_rate=0.0970806646812754#梯度下降学习率

    #构建RNN，开始训练
    rnn = RNNSLU(nh=nhidden,nc=nclasses,ne=vocsize,de=ndim,cs=winsize)
    epoch=0
    while epoch<10:
        shuffle([train_lex, train_ne, train_y], 345)
        for i, (x, y) in enumerate(zip(train_lex, train_y)):
            rnn.train(x, y,winsize,learn_rate)
        print epoch
        epoch+=1
    #测试集的输出标签
    predictions_test = [map(lambda x: idx2label[x],rnn.classify(numpy.asarray(contextwin(x,winsize)).astype('int32'))) for x in test_lex]
    #测试集的正确标签、及其对应的文本
    groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]

    words_test = [map(lambda x: idx2word[x], w) for w in test_lex]
    conlleval(predictions_test,groundtruth_test,words_test, 'current.test.txt')



#main()
input_text =Segment('Data/msr_trainingtemp.utf8')
x,y=input_text.encode_index()
word=input_text.decode_index(x)

for i,j in zip(word,y):
    print i,j