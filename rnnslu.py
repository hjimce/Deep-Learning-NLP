#coding=utf-8
from collections import OrderedDict
import os
import random
import numpy
import theano
from theano import tensor as T
from Preprocess.LoadData import  Segment,write
import numpy as np



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

        self.lastlabel=theano.shared(name='lastlabel',value=0.2 * numpy.random.uniform(-1.0, 1.0,(nc, nc)).astype(theano.config.floatX))
        self.prelabel=theano.shared(name='prelabel',value=0.2 * numpy.random.uniform(-1.0, 1.0,(nc, nc)).astype(theano.config.floatX))
        self.bhmm=theano.shared(name='bhmm',value=numpy.zeros(nc,dtype=theano.config.floatX))

        self.params = [self.emb, self.wx, self.wh, self.w,self.bh, self.b, self.h0,self.lastlabel,self.prelabel,self.bhmm]#所有待学习的参数
        lr = T.scalar('lr')#学习率，一会儿作为输入参数



        idxs = T.itensor3()
        x = self.emb[idxs].reshape((idxs.shape[0],idxs.shape[1],de*idxs.shape[2]))
        y_sentence = T.imatrix('y_sentence')  # 训练样本标签,二维的(batch,sentence)
        def step(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)#通过ht-1、x计算隐藏层
            s_temp=T.dot(h_t, self.w) + self.b#由于softmax不支持三维矩阵操作，所以这边需要对其进行reshape成2D，计算完毕后再reshape成3D
            return h_t, s_temp
        [h,s_temp], _ = theano.scan(step,sequences=x,outputs_info=[T.ones(shape=(x.shape[1],self.h0.shape[0])) * self.h0, None])
        p_y =T.nnet.softmax(T.reshape(s_temp,(s_temp.shape[0]*s_temp.shape[1],-1)))
        p_y=T.reshape(p_y,s_temp.shape)

        #加入前一时刻的标签约束项
        y_label3d = T.ftensor3('y_sentence3d')
        p_ytrain=self.add_layer(p_y,y_label3d)
        loss=self.nll_multiclass(p_ytrain,y_sentence)+0.0*((self.wx**2).sum()+(self.wh**2).sum()+(self.w**2).sum())
        #神经网络的输出
        sentence_gradients = T.grad(loss, self.params)
        sentence_updates = OrderedDict((p, p - lr*g) for p, g in zip(self.params, sentence_gradients))
        self.sentence_traintemp = theano.function(inputs=[idxs,y_sentence,y_label3d,lr],outputs=loss,updates=sentence_updates)





        '''self.sentence_train = theano.function(inputs=[idxs,y_sentence,lr],outputs=loss,updates=sentence_updates)'''
        #词向量归一化，因为我们希望训练出来的向量是一个归一化向量
        self.normalize = theano.function(inputs=[],updates={self.emb:self.emb /T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0, 'x')})

        #构造预测函数、训练函数，输入数据idxs每一行是一个样本(也就是一个窗口内的序列索引)
        #)
        self.classify = theano.function(inputs=[idxs], outputs=p_y)
    def add_layer(self,pre_y,y_label3d):
        s=theano.tensor.as_tensor_variable(T.arange(T.shape(pre_y)[1]))
        p_y_hmm_temp=T.dot(pre_y[:,s,:],self.lastlabel)+T.dot(y_label3d[:,s-1,:],self.prelabel)+self.bhmm
        p_y_hmm =T.nnet.softmax(T.reshape(p_y_hmm_temp,(p_y_hmm_temp.shape[0]*p_y_hmm_temp.shape[1],-1)))
        p_y_hmm=T.reshape(p_y_hmm,p_y_hmm_temp.shape)
        return  p_y_hmm
    #训练
    def train(self, x, y,y3d,learning_rate):
        loss=self.sentence_traintemp(x, y,y3d,learning_rate)
        self.normalize()
        return  loss
    def nll_multiclass(self,p_y_given_x, y):
        p_y =p_y_given_x
        p_y_m = T.reshape(p_y, (p_y.shape[0] * p_y.shape[1], -1))
        y_f = y.flatten(ndim=1)
        return -T.mean(T.log(p_y_m)[T.arange(p_y_m.shape[0]), y_f])

    #保存、加载训练模型
    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())
    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))
#为了采用batch训练，需要保证每个句子长度相同，因此这里采用均匀切分，不过有一个缺陷那就是有可能某个词刚好被切开
def convert2batch(dic,filename,win,length=3):
    x,y=dic.encode_index(filename)#创建训练数据的索引序列
    x2,y2=dic.encode_index('Data/msr/pku_training.utf8')#创建训练数据的索引序列
    x3,y3=dic.encode_index('Data/msr/1.txt')
    x4,y4=dic.encode_index('Data/msr/1998.txt')
    x=x+x2+x3+x4
    y=y+y2+y3+y4


    train_batchxs=[]
    train_batchys=[]




    train_seqx=[x[i:i+length] for i in range(len(x)) if i%length==0]
    train_seqy=[y[i:i+length] for i in range(len(y)) if i%length==0]
    for x,y in zip(train_seqx,train_seqy):
        if len(x)!=length or len(y)!=length:
            continue
        s=contextwin(x,win)
        train_batchxs.append(s)
        train_batchys.append(y)

    #每个句子的长度不同，不能直接转换
    return  np.asarray(train_batchxs,dtype=np.int32),np.asarray(train_batchys,dtype=np.int32)
def add_layer_class(rnn,pre_y):
    result=np.zeros((pre_y.shape[0],pre_y.shape[1]))
    s=np.shape(pre_y)[1]
    y_0 = np.argmax(pre_y[:,0,:],axis=-1)
    result[:,0]=y_0
    lasty=np.zeros((pre_y.shape[0],4),dtype=np.float32)
    lasty[:,y_0]=1
    for i in range(s)[1:]:
        p_y_hmm_temp=np.dot(pre_y[:,i,:],rnn.lastlabel.get_value())+np.dot(lasty,rnn.prelabel.get_value())+rnn.bhmm.get_value()
        y_0=np.argmax(p_y_hmm_temp,axis=-1)
        result[:,i]=y_0
        lasty=np.zeros((pre_y.shape[0],4),dtype=np.float32)
        lasty[:,y_0]=1
        #p_y_hmm=T.reshape(p_y_hmm,p_y_hmm_temp.shape)
    return  result

#RNN分词
def segment_train(dic,filename):

    winsize=5#窗口大小
    trainx,trainy=convert2batch(dic,filename,winsize,1000)

    '''for i,j in zip(train_xbatchs,train_ybatchs):
        for ui,uj in zip(i,j):
            print train_data.decode_index([ui])[0],train_data.decode_index([uj],is_label=True)'''

    '''label=train_data.decode_index(train_y,is_label=True)
    for i,j in zip(word,label):
        print i,j
    #write(word,'label.txt')
    #write(y,'label2.txt')'''


    #计算相关参数
    vocsize = len(dic.word2index)#计算词的个数
    nclasses =len(dic.label2index)#标签数为B、M、E、S

    ndim=50#词向量维度
    nhidden=200#隐藏层的神经元个数
    learn_rate=0.5#梯度下降学习率

    #构建RNN，开始训练
    rnn = RNNSLU(nh=nhidden,nc=nclasses,ne=vocsize,de=ndim,cs=winsize)

    batch_size=64
    n_train_batch=trainx.shape[0]/batch_size
    rnn.load('model/')
    trainy3D=np.zeros((trainy.shape[0],trainy.shape[1],nclasses),dtype=np.float32)
    for i in range(trainy.shape[0]):
        for j in range(trainy.shape[1])[:-1]:
            trainy3D[i,j,trainy[i,j]]=1.
    epoch=0
    while epoch<20:
        shuffle([trainx,trainy,trainy3D], 345)
        loss=0
        for i in range(n_train_batch):
                batx=trainx[i*batch_size:(i+1)*batch_size]
                baty=trainy[i*batch_size:(i+1)*batch_size]
                baty3d=trainy3D[i*batch_size:(i+1)*batch_size]
                decay_lr=learn_rate*0.5**(epoch/50)
                #loss+=rnn.train(batx,baty,decay_lr)
                loss+=rnn.train(batx,baty,baty3d,decay_lr)
                #test=rnn.classify(batx)

                #print
        print 'epoch:',epoch,'\tloss:',loss/n_train_batch
        epoch+=1
    rnn.save('model/')
    return  rnn
def segment_test(model,dic,test_file):
    model.load('model/')#加载训练参数
    x,y=dic.encode_index(test_file)#创建训练数据的索引序列
    xjieba,yjieba=dic.encode_index('Data/msr/msr_test_jieba_result.txt')#创建训练数据的索引序列
    test_batchxs=[]
    test_batchys=[]
    s=contextwin(x,5)
    test_batchxs.append(s)
    test_batchys.append(y)

    test_batchxs=np.asarray(test_batchxs)
    print test_batchxs.shape
    pre=model.classify(test_batchxs)
    pre=add_layer_class(model,pre)
    print pre.shape


    #测试集的输出标签

    predictions_label=dic.decode_index(pre[0],is_label=True)
    groudtrue_label=dic.decode_index(y,is_label=True)
    words_test=dic.decode_index(x)


    for k,(w,i,j) in enumerate(zip(words_test,groudtrue_label,predictions_label)):
        print w,i,j

    #测试集的正确标签、及其对应的文本
    #groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]

    #words_test = [map(lambda x: idx2word[x], w) for w in test_lex]
    print 'save'
    conlleval(predictions_label,groudtrue_label,words_test, 'current.test.txt')


dic=Segment('Data/msr/msr_training.utf8')#创建词典
model=segment_train(dic,'Data/msr/msr_training.utf8')
segment_test(model,dic,'Data/msr/msr_test_gold.utf8')
