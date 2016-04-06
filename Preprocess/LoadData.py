#coding=utf-8
import re
import cPickle
import os
import codecs
def loadfile(filename):
    file=codecs.open(filename,'r','utf-8')
    text=file.read()
    return  text




