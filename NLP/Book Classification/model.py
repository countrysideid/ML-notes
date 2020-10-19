'''
Author: xiaoyao jiang
LastEditors: xiaoyao jiang
Date: 2020-08-31 14:19:30
LastEditTime: 2020-08-31 15:09:29
FilePath: /newBookClassification/model.py
Desciption:  
'''
import json
import jieba
import joblib
import lightgbm as lgb
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import BinaryRelevance

from embedding import Embedding
from features import (get_basic_feature, get_embedding_feature,
                      get_lda_features, get_tfidf)


class BookClassifier:
    def __init__(self, train_mode=False) -> None:
        self.stopWords = [
            x.strip() for x in open('./data/stopwords.txt').readlines()
        ]
        self.embedding = Embedding()
        self.embedding.load()
        self.labelToIndex = json.load(
            open('./data/label2id.json', encoding='utf-8'))
        self.ix2label = {v: k for k, v in self.labelToIndex.items()}
        if train_mode:
            self.train = pd.read_csv('./data/train.csv',
                                     sep='\t').dropna().reset_index(drop=True)
            self.dev = pd.read_csv('./data/dev.csv',
                                   sep='\t').dropna().reset_index(drop=True)
            self.test = pd.read_csv('./data/test.csv',
                                    sep='\t').dropna().reset_index(drop=True)
        self.exclusive_col = ['_id', 'text', 'title', 'content', 'firstClass', 'secondClass', 'lda', 'bow', 'label', 'desc']

    def feature_engineer(self, data):
        data = get_tfidf(self.embedding.tfidf, data)
        data = get_embedding_feature(data, self.embedding.w2v)
        data = get_lda_features(data, self.embedding.lda)
        data = get_basic_feature(data)
        return data

    def trainer(self):
        train = self.feature_engineer(self.train)
        dev = self.feature_engineer(self.dev)
        cols = [x for x in self.train.columns if x not in self.exclusive_col]
        X_train = train[cols]
        y_train = train['label'].apply(lambda x: eval(x))

        X_test = dev[cols]
        y_test = dev['label'].apply(lambda x: eval(x))

        mlb = MultiLabelBinarizer(sparse_output=False)
        y_train = mlb.fit_transform(y_train)
        y_test = mlb.transform(y_test)

        self.clf_BR = BinaryRelevance(classifier=lgb.LGBMClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            silent=True,
            objective='binary',
            nthread=-1,
            reg_alpha=0,
            reg_lambda=1,
            device='gpu',
            missing=None),
                                      require_dense=[False, True])
        self.clf_BR.fit(X_train, y_train)
        prediction = self.clf_BR.predict(X_test)
        print(metrics.accuracy_score(y_test, prediction))

    def save(self):
        joblib.dump(self.clf_BR, './model/clf_BR')

    def load(self):
        self.model = joblib.load('./model/clf_BR')

    def predict(self, title, desc):
        df = pd.DataFrame([[title, desc]], columns=['title', 'desc'])
        df['text'] = df['title'] + df['desc']
        df['text'] = df['text'].apply(lambda x: " ".join(
            [w for w in jieba.cut(x) if w not in self.stopWords and w != '']))
        df = get_tfidf(self.embedding.tfidf, df)
        df = get_embedding_feature(df, self.embedding.w2v)
        df = get_lda_features(df, self.embedding.lda)
        df = get_basic_feature(df)
        cols = [x for x in df.columns if x not in self.exclusive_col]
        pred = self.model.predict(df[cols]).toarray()[0]
        return [self.ix2label[i] for i in range(len(pred)) if pred[i] > 0]


if __name__ == "__main__":
    bc = BookClassifier(train_mode=True)
    bc.trainer()
    bc.save()