# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# IMPORT DATA 

data = pd.read_csv(r'C:\Users\edd_3\OneDrive\Escritorio\Coursera\data_analysis_python\semana_3\mushrooms.csv')
#print(data)

sns.relplot(data = data, x = 'cap-color', y = 'cap-shape', hue = 'class')
plt.show()

# Convert data in letter to numbers 

from sklearn.preprocessing import LabelEncoder

last = len(data.columns)
X = data[data.columns[1:last]]
y = data[data.columns[0]]

#print(y)
le = LabelEncoder()
y = le.fit_transform(y)
#print(y)

for i in X:
    X[i] = le.fit_transform(X[i])
    #print(X[i])

#print(X)

# SPLIT AND TRAIN MODEL

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# TRAINING MODEL 

model = svm.SVC(kernel = 'linear')
classifier = model.fit(X_train, y_train)
y_predict = classifier.predict(X_test)

# EVALUATE RESULT

print(confusion_matrix(y_test, y_predict))
print(accuracy_score(y_test,y_predict))

# FIDIND THE BEST PARAMETERS FOR THIS CASE

def kernel_method(c):
  classifier = svm.SVC(C = c)
  classifier.fit(X_train, y_train)

  y_pred = classifier.predict(X_test)
  accuracy = accuracy_score(y_test,y_pred)
  return accuracy

svm_results = pd.DataFrame({'C':np.arange(1, 10)})
svm_results['Accuracy'] = svm_results['C'].apply(kernel_method)
print(svm_results)


# Evaluando los 3 tipos de criterion con sus respectivos Depth

# Create a list of splitting criteria
splitting_criteria = ['linear', 'poly', 'rbf']

# Create a list of C values
c_values = range(1, 20, 2)

# Create an empty dictionary to store the results
results = {}

# Train Decision Tree models with different splitting criteria and max depth
for criterion in splitting_criteria:
    for c in c_values:
        dt_model = svm.SVC(C = c, kernel = str(criterion))
        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[(criterion, c)] = {
            'model': dt_model,
            'accuracy': accuracy
        }

print("Results of Decision Tree Classification:")
for (criterion, c), result in results.items():
    print(f"Criterion: {criterion}, C: {c}, Accuracy: {result['accuracy']:.2f}")
