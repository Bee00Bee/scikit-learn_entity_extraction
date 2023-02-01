#%%
import itertools
import numpy as np
import pandas as pd
import nltk
import sklearn as sk
import sklearn_crfsuite as skc
from sklearn.metrics import classification_report
import eli5

nltk.download('conll2002')

train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
print(train_sents[0])
'''
train_sents has list of lists. each list has the tokens of a sentenece. For example:
    
    ['Melbourne', 'NP', 'B-LOC'], ['(', 'Fpa', 'O'], ['Australia', 'NP', 'B-LOC'], [')', 'Fpt', 'O'] 
    
the first element in each sublist is a word or token, 
the second element is postag,
and the third element is lable.
'''
train_lst = []
test_lst = []

for s in train_sents:
    for i, j, k in s:
        train_lst.append((len(s), i, j, k))

for s in test_sents:
    for i, j, k in s:
        test_lst.append((len(s), i, j, k))

df_train = pd.DataFrame(train_lst, columns=['sent_id', 'token', 'entity', 'tag'])

df_test = pd.DataFrame(test_lst, columns=['sent_id', 'token', 'entity', 'tag'])

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

print(X_train[0][1])

crf = skc.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)
crf.fit(X_train, y_train);

all_tags = sorted(df_train['tag'].unique().tolist())

y_pred = crf.predict(X_test)

y_test_flat = list(itertools.chain.from_iterable(y_test))
y_pred_flat = list(itertools.chain.from_iterable(y_pred))


results = classification_report(y_test_flat, y_pred_flat, target_names=list(set(y_test_flat)))
print(results)

'''
let's check the predicted labels with real labels from test_sents
'''
df_test['predicted_tags'] = y_pred_flat

print(display(df_test.head(4)))