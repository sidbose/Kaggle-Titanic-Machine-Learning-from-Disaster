import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer # to be used for dealing with NaN values in the data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder # for categorical data

# loading traing and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

drop_features = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train_data.drop(drop_features, axis=1, inplace=True)
test_data.drop(drop_features, axis=1, inplace=True)

train_data['Embarked'].fillna(train_data['Embarked'].value_counts().index[0], inplace=True)
X = train_data.iloc[:, 1:].values
y = train_data.iloc[:, 0:1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:6])
X[:, 2:6] = imputer.transform(X[:, 2:6])

# encoding categorical features
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
onehotencoder = OneHotEncoder(categorical_features=[1, 6])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# selecting SVC model
from sklearn.svm import SVC
#model.fit(X_trai n, y_train)
# Applying Grid Search to find the best model and the best parameters
# from sklearn.model_selection import GridSearchCV
# parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#               {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': np.arange(0.1, 0.9, 0.5)}]
# grid_search = GridSearchCV(estimator=SVC(),
#                            param_grid=parameters,
#                            scoring='accuracy',
#                            cv=10,
#                            n_jobs=-1,
#                            verbose=10)
# c,l=y_train.shape
# grid_search.fit(X_train, y_train.reshape(c,))
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print(best_accuracy)
# print(best_parameters)

model = SVC(C=1, gamma=0.10000000000000001, kernel='rbf', random_state=0)
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy
from sklearn.metrics import accuracy_score
print("SVM model Accuracy: ", accuracy_score(y_test, y_pred))


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
c,l = y_train.shape
classifier.fit(X_train, y_train.reshape(c, ))

# Predicting the Test set results
y_pred_3 = classifier.predict(X_test)

# Accuracy
print("K-NN Accuracy: ", accuracy_score(y_test, y_pred_3))

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=30, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train.reshape(c, ))

# Predicting the Test set results
y_pred_4 = classifier.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
print("Random forest Accuracy: ", accuracy_score(y_test, y_pred_4))


'''
    Neural network implementation
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
classifier = Sequential()
classifier.add(Dense(output_dim=500, init='uniform', activation='relu', input_dim=10))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim=500, init='uniform', activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim=500, init='uniform', activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, nb_epoch=300)

y_predict = classifier.predict(X_test)
y_predict = (y_predict > 0.5).astype(int)

# Accuracy
from sklearn.metrics import accuracy_score
print("Neural Network accuracy: ", accuracy_score(y_test, y_predict))







