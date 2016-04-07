#coding=utf-8
import codecs
#读取中文utf8格式文本文件
def loadfile(filename):
    file=codecs.open(filename,'r','utf-8')
    text=file.read()
    return  text.split()#去除空格
#根据文本input_text生成字典，并返回字典每个字对应的索引、input_text的对应序列索引
def encode(input_text):
    word2index={}
    index=0
    out_putindex=[]
    for a in input_text:
        if  not word2index.has_key(a):
            word2index[a]=index
            index+=1
        out_putindex.append(word2index[a])
    return  out_putindex,word2index#字典的格式为{“中国”：index}
#解码，根据字典、文本句子索引序列，解码出原始的中文文本
def decode(index_array,word2index):
    out_puttext=[]
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
    #index2word、word2index用于词典索引相互映射
    #input_index、input_label用于存储训练数据filename的：文本对应的索引序列、标签序列
    def __init__(self,filename):
        input_text =loadfile(filename)
        label=seglabel(input_text)
        input_texttemp,self.input_label=split(input_text,label)

        self.input_index,self.word2index=encode(input_texttemp)
        self.index2word=decode(self.input_index,self.word2index)
    def encode_index(self):
        return  self.input_index,self.input_label
    def decode_index(self,index_array):
        out_puttext=[]
        for i in index_array:
            out_puttext.append(self.index2word[i])
        return out_puttext














