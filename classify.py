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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer


"""
Data
"""
csvfile = open('data/names.csv')
reader = csv.reader(csvfile)
names = set([row[0] for row in reader])
csvfile.close()

last_names = np.loadtxt('data/last_names.txt', dtype='str', usecols=0).tolist()
last_names = set([x[0] + x[1:].lower() for x in last_names])

months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 
          'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sept', 'oct', 'nov', 'dec', 'present', 'current']

years = [str(x) for x in range(1980, 2020)] + [str(x) for x in range(80, 100)] + ['0' + str(x) for x in range(10)] + [str(x) for x in range(10, 20)]

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
Preprocessing Functions
"""

def tokenize(filename):
    f = open(filename, 'rb')
    text = ''.join(i for i in f.read() if ord(i)<128)
    text_lines = [i for i in text.split('\n') if i != '']
    text_tokens = [line.split() for line in text_lines]
    tagged_text = [nltk.pos_tag(line) for line in text_tokens]
    f.close()
    return tagged_text


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


def contains_year(line):
    for word in line:
        for year in years:
            if year in word:
                return True
    return False

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
        'contains_digit': contains_digit(word),
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
        'is_month': ''.join(x for x in word.lower() if x.isalpha()) in months,
        'is_year': ''.join(x for x in word.lower() if x.isdigit()) in years,
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
            '-1:contains_digit': contains_digit(word1),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:word_len': len(word1),
            '-1:is_first_name': word1 in names,
            '-1:is_last_name': word1 in last_names,
            '-1:is_city': word1 in cities,
            '-1:is_school': word1 in schools,
            '-1:is_state': word1 in states,
            '-1:is_month': ''.join(x for x in word1.lower() if x.isalpha()) in months,
            '-1:is_year': ''.join(x for x in word1.lower() if x.isdigit()) in years,
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
            '+1:contains_digit': contains_digit(word1),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:word_len': len(word1),
            '+1:is_first_name': word1 in names,
            '+1:is_last_name': word1 in last_names,
            '+1:is_city': word1 in cities,
            '+1:is_school': word1 in schools,
            '+1:is_state': word1 in states,
            '+1:is_month': ''.join(x for x in word1.lower() if x.isalpha()) in months,
            '+1:is_year': ''.join(x for x in word1.lower() if x.isdigit()) in years,
        })
    else:
        features['EOS'] = True

    return features



def featurize_sent(sent):
    return [featurize_word(sent, i) for i in range(len(sent))]

# returns X_train, a list of lists of dicts and y_train a list of lists
def get_data_and_labels():
    train_sents = pickle.load(open('text.p', 'rb'))
    train_sents = [nltk.pos_tag(line) for line in train_sents]
    X_train = [featurize_sent(s) for s in train_sents]
    y_train = pickle.load(open('labels.p', 'rb'))
    return X_train, y_train


"""
CRF Functions
"""


def train_crf(X_train, y_train):
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


def tune_crf():
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )

    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    labels = ['n', 'c', 'd', 'l']
    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=30,
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)

    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)

def write_crf_predictions(directory):
    for f in os.listdir(directory):
        if '.txt' in f:
            filename = directory + '/' + f
            crf = pickle.load(open('crf_model.p', 'rb'))
            tagged_text = tokenize(filename)
            resume_text = [featurize_sent(s) for s in tagged_text]
            preds = crf.predict(resume_text)
            f = codecs.open(filename, 'rb')
            text_lines = [i for i in f.read().split('\n') if i != '']
            f.close()
            print('****************************')
            print('******** NEW RESUME ********')
            print('****************************')
            print('\n')
            print('FILENAME = ' + filename)
            print('\n')
            for i in range(len(preds)):
                print(text_lines[i] + ' | ' + ' '.join(preds[i]))
            print('\n')
            print('\n')


"""
Random Forest Functions
"""

def train_RF(X_train, y_train, depth=None):
    X_train = list(chain.from_iterable(X_train))
    y_train = list(chain.from_iterable(y_train))
    dv_X = DictVectorizer(sparse=False)
    lb_y = LabelBinarizer()
    X_train = dv_X.fit_transform(X_train)
    y_train = lb_y.fit_transform(y_train)
    if depth != None:
        rf = RandomForestClassifier(max_depth=depth)
    else:
        rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # for p in rf.feature_importances_:
    #     print(p)
    # print(dv_X.get_feature_names())

    with open('rf_model.p', 'w') as f:
        pickle.dump(rf, f)

    with open('dv_X.p', 'w') as f:
        pickle.dump(dv_X, f)

    with open('lb_y.p', 'w') as f:
        pickle.dump(lb_y, f)


def write_rf_predictions(directory):
    for f in os.listdir(directory):
        if '.txt' in f:
            filename = directory + '/' + f
            rf = pickle.load(open('rf_model.p', 'rb'))
            dv_X = pickle.load(open('dv_X.p', 'rb'))
            lb_y = pickle.load(open('lb_y.p', 'rb'))
            tagged_text = tokenize(filename)
            resume_text = dv_X.transform(list(chain.from_iterable([featurize_sent(s) for s in tagged_text])))
            preds = lb_y.inverse_transform(rf.predict(resume_text))
            f = open(filename, 'rb')
            text = ''.join(i for i in f.read() if ord(i)<128)
            text_lines = [i for i in text.split('\n') if i != '']
            print('****************************')
            print('******** NEW RESUME ********')
            print('****************************')
            print('\n')
            print('FILENAME = ' + filename)
            print('\n')
            counter = 0
            for i in range(len(text_lines)):
                cur_line_preds = []
                for word in text_lines[i].split():
                    cur_line_preds += [preds[counter]]
                    counter += 1
                print(text_lines[i] + ' | ' + ' '.join(cur_line_preds))
            print('\n')
            print('\n')

"""
Extracting Functions
"""
def write_bootstrap_predictions(directory, load_old_bootstrapping):
    new_data = []
    new_labels = []
    if load_old_bootstrapping:
        new_data = pickle.load(open('new_text.p', 'rb'))
        new_labels = pickle.load(open('new_labels.p', 'rb'))
    for f in os.listdir(directory):
        if '.txt' in f:
            filename = directory + '/' + f
            rf = pickle.load(open('rf_model.p', 'rb'))
            dv_X = pickle.load(open('dv_X.p', 'rb'))
            lb_y = pickle.load(open('lb_y.p', 'rb'))
            tagged_text = tokenize(filename)
            resume_text = dv_X.transform(list(chain.from_iterable([featurize_sent(s) for s in tagged_text])))
            rf_preds = lb_y.inverse_transform(rf.predict(resume_text))
            crf = pickle.load(open('crf_model.p', 'rb'))
            resume_text = [featurize_sent(s) for s in tagged_text]
            crf_preds = list(chain.from_iterable(crf.predict(resume_text)))
            f = open(filename, 'rb')
            text = ''.join(i for i in f.read() if ord(i)<128)
            text_lines = [i for i in text.split('\n') if i != '']
            print('****************************')
            print('******** NEW RESUME ********')
            print('****************************')
            print('\n')
            print('FILENAME = ' + filename)
            print('\n')
            counter = 0
            for i in range(len(text_lines)):
                cur_line_preds = []
                cur_line_words = []
                cur_line_labels = []
                for word in text_lines[i].split():
                    if rf_preds[counter] == crf_preds[counter]:
                        cur_line_preds += [rf_preds[counter]]
                        cur_line_labels += [rf_preds[counter]]
                        cur_line_words += [list(chain.from_iterable(resume_text))[counter]]
                    else:
                        cur_line_preds += ['-']
                    counter += 1
                new_data += [cur_line_words]
                new_labels += [cur_line_labels]
                print(text_lines[i] + ' | ' + ' '.join(cur_line_preds))
            print('\n')
            print('\n')

    with open('new_text.p', 'w') as f:
        pickle.dump(new_data, f)

    with open('new_labels.p', 'w') as f:
      pickle.dump(new_labels, f)


def locate_dates(text, labels):
    pass



def locate_companies(text, labels, dates):
    pass


"""
CRF Script

uncomment all lines to run and print predictions
"""

# X_train, y_train = get_data_and_labels()
# train_crf(X_train, y_train)
# if len(sys.argv) >= 2:
#     write_crf_predictions(sys.argv[1])


"""
Random Forest Script

uncomment all lines to run and print predictions
"""

# X_train, y_train = get_data_and_labels()
# train_RF(X_train, y_train, 2)
#if len(sys.argv) >= 2:
    #write_rf_predictions(sys.argv[1])
    

"""
Bootstrapping Script
"""
if len(sys.argv) >= 2:
    X_train, y_train = get_data_and_labels()
    load_old_bootstrapping = False
    if len(sys.argv) >= 3 and sys.argv[2] == 'use_bootstrap':
        new_X = pickle.load(open('new_text.p', 'rb'))
        new_y = pickle.load(open('new_labels.p', 'rb'))
        X_train = X_train + new_X
        y_train = y_train + new_y
        load_old_bootstrapping = True
    train_RF(X_train, y_train)
    train_crf(X_train, y_train)
    write_bootstrap_predictions(sys.argv[1], load_old_bootstrapping)













