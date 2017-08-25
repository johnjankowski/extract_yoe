import sys
import os
import nltk
import sklearn
import scipy.stats
import sklearn_crfsuite
import pickle
import codecs
import re
import csv
import numpy as np
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from itertools import chain
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV


"""
Lists for features
"""
csvfile = open('data/names.csv')
reader = csv.reader(csvfile)
names = set([row[0] for row in reader])
csvfile.close()

last_names = np.loadtxt('data/last_names.txt', dtype='str', usecols=0).tolist()
last_names = set([x[0] + x[1:].lower() for x in last_names])

months = set(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 
              'December', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])

states = set(['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
              'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 
              'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi','Missouri', 
              'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 
              'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 
              'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'AL',
              'AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN',
              'MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'])

csvfile = open('data/cities.csv')
reader = csv.reader(csvfile)
cities = set([row[0] for row in reader])
csvfile.close()

csvfile = open('data/schools.csv')
reader = csv.reader(csvfile)
schools = set([row[0] for row in reader])
csvfile.close()

"""
Preprocessing
"""

def tokenize(filename):
    f = open(filename, 'rb')
    text = ''.join(i for i in f.read() if ord(i)<128)
    text_lines = [i for i in text.split('\n') if i != '']
    text_tokens = [line.split() for line in text_lines]
    tagged_text = [nltk.pos_tag(line) for line in text_tokens]
    f.close()
    return tagged_text


# def create_labels(text):
#     all_labels = []
#     for line in text:
#         line_labels = []
#         for word in line:
#             line_labels += [raw_input(str(word) + ": ")]
#         all_labels += line_labels
#     return labels


# with open('labels.p', 'w') as f:
#   labels = create_labels(tokenize('raw_resumes.txt'))
#   pickle.dump(labels, f)

def contains_digit(word):
    return any(char.isdigit() for char in word)

def is_camel_case(word):
    if re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
        return True
    else:
        return False

def is_mixed_case(word):
    if re.match('[A-Za-z]+$', word):
        return True
    else:
        return False

def contains_hyphen(word):
    if re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
        return True
    else:
        return False

# train_sents = [nltk.pos_tag(line) for line in pickle.load(open('text.p', 'rb'))]
# train_labels = pickle.load(open('labels.p', 'rb'))

def featurize_word(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'last_three_letters': word[-3:],
        'last_two_letters': word[-2:],
        'is_uppercase': word.isupper(),
        'is_lower': word.islower(),
        'is_title_format': word.istitle(),
        'is_digit': word.isdigit(),
        'is_cc': is_camel_case(word),
        'is_mixed_case': is_mixed_case(word),
        'contains-hyphen': contains_hyphen(word),
        'postag': postag,
        'postag[:2]': postag[:2],
        'word_len': len(word),
        'is_first_name': word in names,
        'is_last_name': word in last_names,
        'is_city': word in cities,
        'is_school': word in schools,
        'is_state': word in states,
        'is_month': word in months,
        'contains_/': '/' in word,
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:is_lower': word1.islower(),
            '-1:is_cc': is_camel_case(word1),
            '-1:is_mixed_case': is_mixed_case(word1),
            '-1:contains-hyphen': contains_hyphen(word1),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:word_len': len(word1),
            '-1:is_first_name': word1 in names,
            '-1:is_last_name': word1 in last_names,
            '-1:is_city': word1 in cities,
            '-1:is_school': word1 in schools,
            '-1:is_state': word1 in states,
            '-1:is_month': word1 in months,
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
            '+1:is_lower': word1.islower(),
            '+1:is_cc': is_camel_case(word1),
            '+1:is_mixed_case': is_mixed_case(word1),
            '+1:contains-hyphen': contains_hyphen(word1),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:word_len': len(word1),
            '+1:is_first_name': word1 in names,
            '+1:is_last_name': word1 in last_names,
            '+1:is_city': word1 in cities,
            '+1:is_school': word1 in schools,
            '+1:is_state': word1 in states,
            '+1:is_month': word1 in months,
        })
    else:
        features['EOS'] = True

    return features



def featurize_sent(sent):
    return [featurize_word(sent, i) for i in range(len(sent))]


# X_train = [featurize_sent(s) for s in train_sents]
# y_train = train_labels

# X_test = [featurize_sent(s) for s in test_sents]
# y_test = [sent_labels(s) for s in test_sents] 

"""
Training
"""
def train(X_train, y_train):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.564,
        c2=0.0279,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)

    with open('crf_model.p', 'w') as f:
      pickle.dump(crf, f)

# params_space = {
#     'c1': scipy.stats.expon(scale=0.5),
#     'c2': scipy.stats.expon(scale=0.05),
# }

# labels = ['n', 'c', 'd', 'l']
# # use the same metric for evaluation
# f1_scorer = make_scorer(metrics.flat_f1_score,
#                         average='weighted', labels=labels)

# # search
# rs = RandomizedSearchCV(crf, params_space,
#                         cv=3,
#                         verbose=1,
#                         n_jobs=-1,
#                         n_iter=30,
#                         scoring=f1_scorer)
# rs.fit(X_train, y_train)

# print('best params:', rs.best_params_)
# print('best CV score:', rs.best_score_)

"""
Predicting
"""
if len(sys.argv) >= 2:
    for f in os.listdir(sys.argv[1]):
        if '.txt' in f:
            filename = sys.argv[1] + '/' + f
            crf = pickle.load(open('crf_model.p', 'rb'))
            tagged_text = tokenize(filename)
            resume_text = [featurize_sent(s) for s in tagged_text]
            preds = crf.predict(resume_text)
            f = codecs.open(filename, 'rb')
            text_lines = [i for i in f.read().split('\n') if i != '']
            #print(text_lines)
            f.close()
            print(filename)
            for i in range(len(preds)):
                if ('c' in preds[i]) and ('d' in preds[i]):
                    print(text_lines[i] + ' | ' + ' '.join(preds[i]))
                    print('\n')
                elif ('c' in preds[i]) and (i + 1 < len(preds)) and ('d' in preds[i + 1]):
                    print(text_lines[i] + ' | ' + ' '.join(preds[i]))
                    print(text_lines[i + 1]) + ' | ' + ' '.join(preds[i + 1])
                    print('\n')
            print('\n')
            print('\n')



# tagged_text = tokenize('raw_resumes.txt')
# resume_text = [featurize_sent(s) for s in tagged_text]

# preds = crf.predict(resume_text)

# f = codecs.open('raw_resumes.txt', 'rb', 'utf-8')
# text_lines = f.read().split('\n')
# f.close()


# TODO: try getting up to two lines above/below dates if they have 'c' label
# for i in range(len(preds)):
#     if ('c' in preds[i]) and (i + 1 < len(preds)) and ('d' in preds[i + 1]):
#         print(text_lines[i] + ' | ' + ' '.join(preds[i]))
#         print(text_lines[i + 1] + ' | ' + ' '.join(preds[i + 1]))
#         print('\n')












