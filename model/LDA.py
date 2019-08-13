from jieba import posseg

from gensim import corpora,models

import re

#gensim的lda模型简单实现
def get_stop_words(path):
    stop_word_list=[]
    with open(path,encoding='utf-8') as file:
        for line in file.readlines():
            stop_word_list.append(line.strip("\t"))


def clean_text(text,path,filter_pos):
    text=''.join(re.findall(r'[\u2E80-\u9FFF]',text))
    stop_word_list=get_stop_words(path)
    clean_text=[x.word for x in posseg.cut(text) if x.word not in stop_word_list and x.flag not in filter_pos]
    return clean_text

def lda(texts,num_topics):
    dic=corpora.Dictionary(texts)
    corpus=[dic.doc2bow(text) for text in texts]
    # tfidf=models.tfidfmodel(corpus)
    # corpus_tfidf=tfidf[corpus]
    lda=models.LdaModel(corpus,id2word=dic,num_topics=num_topics)
    lda.save('lda.model')
    topics=lda.print_topics(num_topics)
    for topic in topics:
        print(topic)