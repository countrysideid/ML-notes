'''
@Author: xiaoyao jiang
@Date: 2020-04-08 17:22:54
LastEditTime: 2020-08-31 14:32:39
LastEditors: xiaoyao jiang
@Description: train embedding & tfidf & autoencoder
FilePath: /newBookClassification/embedding.py
'''
import pandas as pd
import numpy as np
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import jieba
from gensim.models import LdaMulticore
from features import label2idx
import gensim


class SingletonMetaclass(type):
    '''
    @description: singleton
    '''
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass,
                                    self).__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance


class Embedding(metaclass=SingletonMetaclass):
    def __init__(self):
        '''
        @description: This is embedding class. Maybe call so many times. we need use singleton model.
        In this class, we can use tfidf, word2vec, fasttext, autoencoder word embedding
        @param {type} None
        @return: None
        '''
        # 停止词
        self.stopWords = [x.strip() for x in open('./data/stopwords.txt').readlines()]

    def load_data(self, path):
        '''
        @description:Load all data, then do word segment
        @param {type} None
        @return:None
        '''
        data = pd.read_csv(path, sep='\t')
        data = data.fillna("")
        data = data[data['title'] != '']
        data = data[data['content'].str.len() > 10]
        data["text"] = data['title'] + data['desc']
        data["text"] = data['text'].apply(lambda x: " ".join([w for w in jieba.cut(x) if w not in self.stopWords and w != '']))
        data['text'] = data["text"].apply(lambda x: " ".join(x))
        self.labelToIndex = label2idx(data)
        data['firstClass'] = data['firstClass'].map(self.labelToIndex)
        data['secondClass'] = data['secondClass'].map(self.labelToIndex)
        data['label'] = data.apply(lambda row: [float(row['firstClass']), float(row['secondClass'])], axis=1)
        data = data[['_id', 'text', 'title', 'content', 'label']]
        self.train, _, _ = np.split(data[['_id', 'text', 'title', 'content', 'label']].sample(frac=1), [int(data.shape[0] * 0.7), int(data.shape[0] * 0.9)])

    def trainer(self):
        '''
        @description: Train tfidf,  word2vec, fasttext and autoencoder
        @param {type} None
        @return: None
        '''
        count_vect = TfidfVectorizer(stop_words=self.stopWords,
                                     max_df=0.4,
                                     min_df=0.001,
                                     ngram_range=(1, 2))
        self.tfidf = count_vect.fit(self.train["text"])

        self.train['text'] = self.train["text"].apply(lambda x: x.split(' '))
        self.w2v = models.Word2Vec(min_count=2,
                                   window=5,
                                   size=300,
                                   sample=6e-5,
                                   alpha=0.03,
                                   min_alpha=0.0007,
                                   negative=15,
                                   workers=4,
                                   iter=30,
                                   max_vocab_size=50000)
        self.w2v.build_vocab(self.train["text"])
        self.w2v.train(self.train["text"],
                       total_examples=self.w2v.corpus_count,
                       epochs=15,
                       report_delay=1)

        self.id2word = gensim.corpora.Dictionary(self.train['text'])
        corpus = [self.id2word.doc2bow(text) for text in self.train['text']]
        self.LDAmodel = LdaMulticore(corpus=corpus,
                                     id2word=self.id2word,
                                     num_topics=30,
                                     workers=4,
                                     chunksize=4000,
                                     passes=7,
                                     alpha='asymmetric')

    def saver(self):
        '''
        @description: save all model
        @param {type} None
        @return: None
        '''
        joblib.dump(self.tfidf, './model/embedding/tfidf')

        self.w2v.wv.save_word2vec_format('./model/embedding/w2v.bin',
                                         binary=False)

        self.LDAmodel.save('./model/embedding/lda')

    def load(self):
        '''
        @description: Load all embedding model
        @param {type} None
        @return: None
        '''
        self.tfidf = joblib.load('./model/tfidf_model')

        self.w2v = models.KeyedVectors.load_word2vec_format('./model/w2v_model', binary=True)

        self.lda = models.ldamodel.LdaModel.load('./model/lda')


if __name__ == "__main__":
    em = Embedding()
    em.load_data('/home/user10000253/dataset/bookClassification/2bd3ad04-cc2d-49ab-9b20-c7c412338122/file/book.csv')
    em.trainer()
    em.saver()
