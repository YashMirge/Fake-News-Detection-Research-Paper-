
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

df = pd.read_csv('C:/Users/Yash/Desktop/Fake News/fake_or_real_news.csv')
print(df.head(10));

print(df.shape);

print(df.columns);

df.drop_duplicates(inplace = True)

print(df.shape);

df.loc[df["label"]=="FAKE","label"]=0
df.loc[df["label"]=="REAL","label"]=1

print(df.head(10));

print(df.isnull().sum());

nltk.download('stopwords')

def process_text(text):
    '''
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    '''
    
    #1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #3
    return clean_words

#Show the Tokenization 
print(df['text'].head().apply(process_text) );

from sklearn.feature_extraction.text import CountVectorizer
News = CountVectorizer(analyzer=process_text).fit_transform(df['text'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(News, df['label'], test_size = 0.40, random_state = 0)

print(News.shape);

print("================================== Naive Bayes ==================================");
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

#Print the predictions
print(classifier.predict(X_train))
#Print the actual values
print(y_train.values)
#Evaluate the model on the training data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_train)
print(classification_report(y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred))

#Print the predictions
print('Predicted value: ',classifier.predict(X_test))
#Print Actual Label
print('Actual value: ',y_test.values)
#Evaluate the model on the test data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_test)
print(classification_report(y_test ,pred ))

print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))
print("===================================================================================");
print('');
print("================================== Random Forest ==================================");
from sklearn.ensemble import RandomForestClassifier

clf1 = RandomForestClassifier(random_state=1)
clf1.fit(X_train, y_train)
#Print the predictions
print(clf1.predict(X_train))
#Print the actual values
print(y_train.values)
#Evaluate the model on the training data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = clf1.predict(X_train)
print(classification_report(y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred))
#Print the predictions
print('Predicted value: ',clf1.predict(X_test))
#Print Actual Label
print('Actual value: ',y_test.values)
#Evaluate the model on the test data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = clf1.predict(X_test)
print(classification_report(y_test ,pred ))

print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))
print("===================================================================================");
print('');
print("================================== Random Forest ==================================");
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#descision tree
dt = DecisionTreeClassifier(criterion='gini',max_depth=None)
dt.fit(X_train,y_train)
# dt.score(X_test,y_test)
#Print the predictions
print(dt.predict(X_train))
# dt.score(X_train,y_train)
#Print the actual values
print(y_train.values)

predictions = dt.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

cm=confusion_matrix(y_test,predictions)
print(cm)
print ("Accuracy of prediction:",accuracy_score(y_test,pred))

#Evaluate the model on the training data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = dt.predict(X_train)
print(classification_report(y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred))

#Print the predictions
print('Predicted value: ',dt.predict(X_test))

#Print Actual Label
print('Actual value: ',y_test.values)

#Evaluate the model on the test data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = dt.predict(X_test)
print(classification_report(y_test ,pred ))

print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))
print("===================================================================================");
print('');
print("================================== Bagging ==================================");
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
bg.fit(X_train,y_train)

print("Test dataset :",bg.score(X_test,y_test));

print("Train dataset : ",bg.score(X_train,y_train));

print("===================================================================================");
print('');
print("================================== Boosting ==================================");
#Boosting - Ada Boost

adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)
adb.fit(X_train,y_train)

print("Test dataset : ",adb.score(X_test,y_test));

print("train dataset : ",adb.score(X_train,y_train));
print("===================================================================================");
print('');
print("================================== Stacking ==================================");

from mlxtend.classifier import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.simplefilter('ignore')

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = MultinomialNB()
lr   = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                          meta_classifier=lr)


print('4-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3,lr, sclf], 
                      ['KNN', 
                       'Random Forest', 
                       'Naive Bayes',
                       'Logistic Regression','StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X_train, y_train,cv=4, scoring='accuracy')
    print("Train dataset","Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


print('4-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3,lr, sclf], 
                      ['KNN', 
                       'Random Forest', 
                       'Naive Bayes',
                       'Logistic Regression','StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X_test, y_test,cv=4, scoring='accuracy')
    print("Test Dataset","Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

print("===================================================================================");
print('');
print("================================== bagging , Boosting and  stacking algorithm accuracy ==================================");
print("bagging ",bg.score(X_test,y_test)) # bagging

print("Boosting ",adb.score(X_test,y_test))  # Boosting

# Stacking
print('4-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3,lr, sclf], 
                      ['KNN', 
                       'Random Forest', 
                       'Naive Bayes',
                       'Logistic Regression','StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X_test, y_test,cv=4, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))