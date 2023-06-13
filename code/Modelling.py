import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import TruncatedSVD 

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, auc, roc_curve, roc_auc_score, RocCurveDisplay

import matplotlib.pyplot as plt

from itertools import cycle

data = pd.read_csv('../data/text_data_for_modelling_sample.csv')
data.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)


# Train-validation-test split

data_train1, data_test= train_test_split(data, test_size=0.2)
data_train, data_validation = train_test_split(data_train1, test_size=12630)


# Bag of words feature extraction
count_vectorizer = CountVectorizer()

X_train = count_vectorizer.fit_transform(data_train['Abstract_v2'])
X_valid = count_vectorizer.transform(data_validation['Abstract_v2'])
X_test = count_vectorizer.transform(data_test['Abstract_v2'])

# Using SVD to reduce the features
svd = TruncatedSVD(n_components=50)
X_train_reduced = svd.fit_transform(X_train)
X_valid_reduced = svd.transform(X_valid)
X_test_reduced = svd.transform(X_test)

y_train = data_train['Primary category']
y_valid = data_validation['Primary category']
y_test = data_test['Primary category']

# Training on bag of words
rf = RandomForestClassifier()
rf.fit(X_train_reduced, y_train)
y_pred_rf = rf.predict(X_test_reduced)
accuracy_rf = accuracy_score(y_valid, y_pred_rf)

dt = DecisionTreeClassifier(class_weight='balanced')
dt.fit(X_train_reduced, y_train)
y_pred_dt = dt.predict(X_valid_reduced)
accuracy_dt = accuracy_score(y_valid, y_pred_dt)

svc = SVC(class_weight='balanced')
svc.fit(X_train_reduced, y_train)
y_pred_svc = svc.predict(X_valid_reduced)
accuracy_svc = accuracy_score(y_valid, y_pred_svc)



# TF-IDF feature extraction
tfidfvectorizer = TfidfVectorizer()

X_train = tfidfvectorizer.fit_transform(data_train['Abstract_v2'])
X_valid = tfidfvectorizer.transform(data_validation['Abstract_v2'])
X_test = tfidfvectorizer.transform(data_test['Abstract_v2'])

# Using SVD to reduce the features
svd = TruncatedSVD(n_components=50)
X_train_reduced = svd.fit_transform(X_train)
X_valid_reduced = svd.transform(X_valid)
X_test_reduced = svd.transform(X_test)

y_train = data_train['Primary category']
y_valid = data_validation['Primary category']
y_test = data_test['Primary category']


# Checking the distribution of output class for all the datasets
y_train.value_counts(normalize=True).plot(kind="bar")
y_valid.value_counts(normalize=True).plot(kind="bar")
y_test.value_counts(normalize=True).plot(kind="bar")

# Training different models: namely random forest, decision tree and SVC
rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train_reduced, y_train)
y_pred_rf = rf.predict(X_valid_reduced)
accuracy_rf = accuracy_score(y_valid, y_pred)

dt = DecisionTreeClassifier(class_weight='balanced')
dt.fit(X_train_reduced, y_train)
y_pred_dt = dt.predict(X_valid_reduced)
accuracy_dt = accuracy_score(y_valid, y_pred_dt)

svc = SVC(class_weight='balanced')
svc.fit(X_train_reduced, y_train)
y_pred_svc = svc.predict(X_valid_reduced)
accuracy_svc = accuracy_score(y_valid, y_pred_svc)

# Decision tree classification report
print(classification_report(y_valid, y_pred_dt))

# Decision tree confusion matrix
cm = confusion_matrix(y_test, y_pred_dt, labels=dt.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
disp.plot()


# Random Forest classification report
print(classification_report(y_valid, y_pred_rf))

# Random Forest confusion matrix
cm = confusion_matrix(y_test, y_pred_rf, labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot()


# SVM classification report
print(classification_report(y_valid, y_pred_svc))

# SVM confusion matrix
cm = confusion_matrix(y_test, y_pred_svc, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
disp.plot()

y_pred_test_svc = svc.predict(X_test_reduced)
accuracy_svc = accuracy_score(y_test, y_pred_test_svc)


y_binarized = label_binarize(y_test, classes=['Other', 'cs.AI', 'cs.CL', 'cs.CR', 'cs.CV', 'cs.IT', 'cs.LG',
       'cs.NA', 'cs.RO', 'cs.SY'])

n_classes = y_binarized.shape[1]


# Code for plotting ROC Curve

fpr = dict()
tpr = dict()
roc_auc = dict()
lw=2
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_predprob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    pred_cls = ['Other', 'cs.AI', 'cs.CL', 'cs.CR', 'cs.CV', 'cs.IT', 'cs.LG',
       'cs.NA', 'cs.RO', 'cs.SY'][i]
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(pred_cls, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for test data')
plt.legend(loc="lower right")
plt.show()
