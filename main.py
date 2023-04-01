from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#load the dataset and call it "data"
data = pd.read_csv(r'C:\Users\SC\Desktop\project\spam.csv')

#remove instances with missing values
data = data.dropna()
print("Dataset Loaded...")

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

#select target and feature variables. X = variables, Y = target
dX = data.iloc[:, :-1].values
dY = data[['Class']].values
rX = data.iloc[:, :-1].values
rY = data[['Class']].values


#split into first 1000 as training and remaining as test data, (dX and dY for decision tree, rX an rY for random forest)
dX_train = dX[:1001, :]
dX_test = dX[1001:, :]
dY_train = dY[:1001]
dY_test = dY[1001:]
rX_train = dX[:1001, :]
rX_test = dX[1001:, :]
rY_train = dY[:1001]
rY_test = dY[1001:]

# finding the best max depth of decision tree
maxdepthpool = {'max_depth': range(1, 10)}

# finding the best n_estimators for random forest
testestimators = {'n_estimators': [100, 150, 200]}

# create a test decision tree classifier object
tclf = DecisionTreeClassifier(random_state=27)

# create a test random forest classifier object
trfc = RandomForestClassifier()

# use grid search to find the best n_estimators value
n_grid_search = GridSearchCV(estimator=trfc, param_grid=testestimators, cv=5)
n_grid_search.fit(rX, rY.ravel())

# use grid search to find the best max_depth
m_grid_search = GridSearchCV(tclf, maxdepthpool, cv=5)
m_grid_search.fit(dX, dY)

# print the best max_depth
print('Best max_depth:', m_grid_search.best_params_['max_depth'], "\n")

# print the best n_estimators
print('Best n_estimators:', n_grid_search.best_params_['n_estimators'], "\n")

# create a decision tree classifier object using the best max_depth found
clf = DecisionTreeClassifier(max_depth=m_grid_search.best_params_['max_depth'],random_state=27)

# create a random forest classifier with the best n_estimator found
rfc = RandomForestClassifier(n_estimators=n_grid_search.best_params_['n_estimators'], random_state=42)

# convert rY_train to a 1-dimensional array using ravel()
rY_train = rY_train.ravel()

# train the random forest classifier on the training set
rfc.fit(rX_train, rY_train)

# train the decision tree model on the training set
clf.fit(dX_train, dY_train)

# make predictions on the random forest testing set
rY_pred = rfc.predict(rX_test)

# make predictions on the decision tree testing set
dY_pred = clf.predict(dX_test)


# evaluate the model
print("Decision Tree Accuracy Score", accuracy_score(dY_test, dY_pred), "\n")
print("Random Forest Accuracy Score", accuracy_score(rY_test, rY_pred), "\n")
print("Decision Tree Classification Report: \n", classification_report(dY_test, dY_pred))
print("Random Forest Classification Report: \n", classification_report(rY_test, rY_pred))
print("Decision Tree Confusion Matrix: \n", confusion_matrix(dY_test, dY_pred))
print("Random Forest Confusion Matrix: \n", confusion_matrix(rY_test, rY_pred))

