#coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import codecs
#读取中文utf8格式文本文件
def loadfile(filename):
    file=codecs.open(filename,'r','utf-8')
    text=file.read()
    return  text.split()#去除空格
def write(liststr,filename):
    f=open(filename,'w')
    for s in liststr:
        f.write(s)
    f.close()



#根据文本input_text生成字典，并返回字典每个字对应的索引、input_text的对应序列索引
def encode(input_text):
    word2index={}
    index=0
    for a in input_text:
        if  not word2index.has_key(a):
            word2index[a]=index
            index+=1
    return  word2index#字典的格式为{“中国”：index}
#解码，根据字典、文本句子索引序列，解码出原始的中文文本
def decode(word2index):
    index2word=[-1]*len(word2index)
    for key in word2index:
        index2word[word2index[key]]=key
    return  index2word
def seglabel(input_text):
    label=[]
    for s in input_text:
        if len(s)==1:
            label.append('S')
        elif len(s)==2:
            label.append('BE')
        else:
            stemp='B'+'M'*(len(s)-2)+'E'
            label.append(stemp)
    return  label
def split(input_text,label):
    input_texttemp=[]
    input_labeltemp=[]
    for a,b in zip(input_text,label):
        if len(a)!=len(b):
            print 'error'
        for i,j in zip(a,b):
            input_texttemp.append(i)
            input_labeltemp.append(j)
    return  input_texttemp,input_labeltemp
class Segment:
    #根据文本filename生成词典，index2word、word2index用于词典索引相互映射
    def __init__(self,filename):
        input_text =loadfile(filename)
        input_texttemp=[]
        for s in input_text:
            for i in s:
                input_texttemp.append(i)
        self.word2index=encode(input_texttemp)
        self.index2word=decode(self.word2index)
        self.label2index={'B':0,'M':1,'E':2,'S':3}
        self.index2label={0:'B',1:'M',2:'E',3:'S'}
    #返回file文本生成的索引序列，以及分词对应的标签序列
    def encode_index(self,filename):
        input_text =loadfile(filename)
        label=seglabel(input_text)
        input_texttemp,label=split(input_text,label)
        input_index=[]
        input_label=[]
        for s,l in zip(input_texttemp,label):
            if self.word2index.has_key(s):
                input_index.append(self.word2index[s])
                input_label.append(self.label2index[l])
        return  input_index,input_label
    #更新文本的索引序列，可以生成原始的文本
    def decode_index(self,index_array,is_label=False):
        out_puttext=[]
        if not is_label:
            for i in index_array:
                out_puttext.append(self.index2word[i])
        else:
            for i in index_array:
                out_puttext.append(self.index2label[i])
        return out_puttext
















