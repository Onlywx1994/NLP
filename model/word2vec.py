import tensorflow as tf

import  numpy as np

import pandas as  pd

import gensim


from gensim.models import word2vec


#word2vec的调用api实现

data=word2vec.LineSentence()
model=gensim.models.Word2Vec(sentences=data,size = 128, min_count = 3, iter = 50)
#保存模型
model.wv.save_word2vec_format("",binary = True)

#加载模型
word2vec=gensim.models.KeyedVectors.load_word2vec_format("", binary = True)


#基于Tensorflow的word2vec实现