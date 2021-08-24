import numpy as np
import pandas as pd 
import nltk
import warnings; warnings.simplefilter('ignore')
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('wordnet')


class LemmaTokenizer(object):
    
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, data):
        return [self.wnl.lemmatize(t) for t in word_tokenize(data)]




categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
dataset_train = fetch_20newsgroups(categories = categories)
dataset_test = fetch_20newsgroups(categories = categories)
X = dataset_train.data
y = dataset_test.target

class_names = fetch_20newsgroups(categories = categories).target_names

# Split the data, 70% -> for training and 30% -> for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=53)



# Plot non-normalized confusion matrix
def plot_matrix(classifier, titlecl, X = X_test, Y = y_test, display_labels = class_names):
    titles_options = [("Confusion matrix, without normalization", None),
                    ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                    display_labels=class_names,
                                    cmap=plt.cm.Blues,
                                    normalize=normalize)
        disp.ax_.set_title(title + '\n' + titlecl)

        print(disp.confusion_matrix)

    plt.show()

########################## Naive Bayes ##########################
text_cl_nb = Pipeline([
     ('vect', CountVectorizer(stop_words = 'english')),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
 ])
 

classifier = text_cl_nb.fit(X_train, y_train)

pred = text_cl_nb.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("Naive Bayes: ")
print("accuracy:   %0.3f" % score)
print('Seocond opinion for the accuacy above: ')
predicted = classifier.predict(X_test)
print("accuracy: ",np.mean(predicted == y_test))


# Plot non-normalized confusion matrix
#plot_matrix(classifier, titlecl = "Naive Bayes without lemmatization and ngrams")

# With lemmatization 
text_cl_nb_lem = Pipeline([
     ('vect', CountVectorizer(tokenizer = LemmaTokenizer(), stop_words = 'english')),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
    ])

classifier_with_lem = text_cl_nb_lem.fit(X_train, y_train)

print("\nNaive Bayes with lemmatization: ")
predictedlem = classifier_with_lem.predict(X_test)
print("accuracy: ",np.mean(predictedlem == y_test))

# Plot non-normalized confusion matrix
#plot_matrix(predictedlem, titlecl = 'Naive Bayes with lemmatization')

# With lemmatization and 2grams
text_cl_nb_lem2gram = Pipeline([
     ('vect', CountVectorizer(stop_words = 'english', ngram_range = (2, 2))),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
    ])

classifier_with_lem2gram = text_cl_nb_lem2gram.fit(X_train, y_train)

print("\nNaive Bayes with 2grams: ")
predicted2gram = classifier_with_lem2gram.predict(X_test)
print("accuracy: ",np.mean(predicted2gram == y_test))

# Plot non-normalized confusion matrix
#plot_matrix(predicted2gram, titlecl = 'Naive Bayes with lemmatization and 2grams')

# With lemmatization and 3grams

text_cl_nb_lem3gram = Pipeline([
     ('vect', CountVectorizer(stop_words = 'english', ngram_range = (3, 3))),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
    ])

classifier_with_lem3gram = text_cl_nb_lem3gram.fit(X_train, y_train)
print("\nNaive Bayes with 3grams: ")
predicted3gram = classifier_with_lem3gram.predict(X_test)
print("accuracy: ",np.mean(predicted3gram == y_test))

# Plot non-normalized confusion matrix
# plot_matrix(predicted3gram, titlecl = 'Naive Bayes with 3grams')


########################## Support vector machines ##########################

# Without Lemmatizatin and ngams
svm_text_clf = Pipeline([
     ('vect', CountVectorizer(stop_words = 'english')),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)),
 ])


SVMclassifier = svm_text_clf.fit(X_train, y_train)
print("\nSupport vector machine: ")
predictedsvm = SVMclassifier.predict(X_test)
print("accuracy: ",np.mean(predictedsvm == y_test))

# Plot non-normalized confusion matrix
#plot_matrix(predicted3gram, titlecl = 'Support vector machine')

# Plot non-normalized confusion matrix
plot_matrix(SVMclassifier, titlecl = 'Support vector machine')
# With lemmatization 
svm_text_clf_lem = Pipeline([
     ('vect', CountVectorizer(tokenizer = LemmaTokenizer(), stop_words = 'english')),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)),
 ])


SVMclassifier_lema = svm_text_clf_lem.fit(X_train, y_train)
print("\nSupport vector machine with lemmatization: ")
predictedsvm_lema = SVMclassifier_lema.predict(X_test)
print("accuracy: ",np.mean(predictedsvm_lema == y_test))

#  2gams
svm_text_clf_n2 = Pipeline([
     ('vect', CountVectorizer(stop_words = 'english', ngram_range = (2, 2))),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)),
 ])


SVMclassifier_n2 = svm_text_clf_n2.fit(X_train, y_train)
print("\nSupport vector machine wiht 2grams: ")
predictedsvm2 = SVMclassifier_n2.predict(X_test)
print("accuracy: ",np.mean(predictedsvm2 == y_test))



# 3gams
svm_text_clf_n3 = Pipeline([
     ('vect', CountVectorizer(stop_words = 'english', ngram_range = (3, 3))),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)),
 ])


SVMclassifier_n3 = svm_text_clf_n2.fit(X_train, y_train)
print("\nSupport vector machine wiht 3grams: ")
predictedsvm3 = SVMclassifier_n3.predict(X_test)
print("accuracy: ",np.mean(predictedsvm3 == y_test))



########################## Random Forest Classifier ##########################
"""
X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
"""

rm_text_clf = Pipeline([
     ('vect', CountVectorizer(stop_words = 'english')),
     ('tfidf', TfidfTransformer()),
     ('clf', RandomForestClassifier(n_estimators=1000, random_state=0))
 ])


RMclassifier = rm_text_clf.fit(X_train, y_train)
print("\nRandom Forest Classifier: ")
predictedrm = RMclassifier.predict(X_test)
print("accuracy: ", np.mean(predictedrm == y_test))


# With lemmatization 

rm_text_clf_lema = Pipeline([
     ('vect', CountVectorizer(tokenizer = LemmaTokenizer(), stop_words = 'english')),
     ('tfidf', TfidfTransformer()),
     ('clf', RandomForestClassifier(n_estimators=1000, random_state=0))
 ])


RMclassifier_lema = rm_text_clf_lema.fit(X_train, y_train)
print("\nRandom Forest Classifier with lemmatization: ")
predictedrm_lema = RMclassifier_lema.predict(X_test)
print("accuracy: ", np.mean(predictedrm_lema == y_test))


# 2grams

rm_text_clf_lem_n2 = Pipeline([
     ('vect', CountVectorizer(stop_words = 'english',ngram_range = (2, 2))),
     ('tfidf', TfidfTransformer()),
     ('clf', RandomForestClassifier(n_estimators=1000, random_state=0))
 ])


RMclassifierlem_n2 = rm_text_clf_lem_n2.fit(X_train, y_train)
print("\nRandom Forest Classifier 2grams: ")
predictedrmlem_n2 = RMclassifierlem_n2.predict(X_test)
print("accuracy: ", np.mean(predictedrmlem_n2 == y_test))
8
#  3grams
rm_text_clf_lem_n2 = Pipeline([
     ('vect', CountVectorizer(stop_words = 'english',ngram_range = (3, 3))),
     ('tfidf', TfidfTransformer()),
     ('clf', RandomForestClassifier(n_estimators=1000, random_state=0))
 ])


RMclassifierlem_n3 = rm_text_clf_lem_n2.fit(X_train, y_train)
print("\nRandom Forest Classifier with 3grams: ")
predictedrmlem_n3 = RMclassifierlem_n3.predict(X_test)
print("accuracy: ", np.mean(predictedrmlem_n3 == y_test))

# Plot non-normalized confusion matrix
plot_matrix(RMclassifierlem_n3, titlecl = 'Random Forest with 3grams')
