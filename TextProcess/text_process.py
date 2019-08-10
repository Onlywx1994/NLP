
import pandas as pd
from collections import Counter

class Process(object):


    def __init__(self,path,min_len,vocab):
        self.path=path
        self.min_len=min_len
        self.vocab=vocab


    def read_data(self):
        with open(self.path,"r") as file:
            text=file.read()
        return text

    def word_count(self):
        text=self.read_data()
        word_dict=Counter(text.split())
        return word_dict


    def word2id_id2word(self):
        vocab=self.vocab
        word_dict=self.word_count()
        for word,freq in word_dict.most_common():
            if freq>self.min_len:
                vocab.append(word)
        word2id={word:id for id ,word in enumerate(vocab)}

        id2word={id:word for id ,word in enumerate(vocab)}

        return word2id,id2word
