import numpy as np # linear algebra
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for data visualization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM


data = r'C:\Users\annet\PycharmProjects\wine_quality\data\winequality-red.csv'
df = pd.read_csv(data)

col_names = ['fixed_acidity','volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',\
             'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates','alcohol', 'quality']

df.columns = col_names

df.info()
print (df.isnull().values.any())
print (df.isna().sum())



print(round(df.describe(),2))

# print (df.isnull().sum())
# print(round(df.describe(),2))

# plt.rcParams['figure.figsize']=(15, 10)
# df.plot(kind='hist', bins=20, subplots=True, layout=(6,2), sharex=False, sharey=False)
# plt.show()

correlation = df.corr()

correlation['quality'].sort_values(ascending=False)

plt.figure(figsize=(10,8))
plt.title('Correlation of Attributes with quality variable')
a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)
plt.show()

X = df.drop(['quality'], axis=1)


for dataset in X:
    dataset.loc[dataset['fixed_acidity']<= 4.60, 'fixed_acidity'] =0
    dataset.loc[(dataset['fixed_acidity'] > 4.60) & (dataset['fixed_acidity']<=7.10),'fixed_acidity'] =1
    dataset.loc[(dataset['fixed_acidity'] > 7.10) & (dataset['fixed_acidity']<=7.90), 'fixed_acidity'] =2
    dataset.loc[(dataset['fixed_acidity'] > 7.90) & (dataset['fixed_acidity'] <= 9.20), 'fixed_acidity'] =3
    dataset.loc[dataset['fixed_acidity'] > 9.20, 'fixed_acidity'] =4

for dataset in X:
    dataset.loc[dataset['free_sulfur_dioxide']<= 1.00, 'free_sulfur_dioxide'] =0
    dataset.loc[(dataset['free_sulfur_dioxide'] > 1.00) & (dataset['free_sulfur_dioxide']<=7.00),'free_sulfur_dioxide'] =1
    dataset.loc[(dataset['free_sulfur_dioxide'] > 7.00) & (dataset['free_sulfur_dioxide']<=14.00), 'free_sulfur_dioxide'] =2
    dataset.loc[(dataset['free_sulfur_dioxide'] > 14.00) & (dataset['free_sulfur_dioxide'] <= 21.00), 'free_sulfur_dioxide'] =3
    dataset.loc[dataset['free_sulfur_dioxide'] > 21.00, 'free_sulfur_dioxide'] =4
df.head()

y = df['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


cols = X_train.columns
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))





# def test_estimators(X, y, estimators, labels, cv):
#     '''
#     A function for testing multiple estimators.
#     It takes: full train data and target, list of estimators,
#               list of labels or names of estimators,
#               cross validation splitting strategy;
#     And it returns: a DataFrame of table with results of tests
#     '''
#     result_table = pd.DataFrame()
#
#     row_index = 0
#     for est, label in zip(estimators, labels):
#         est_name = label
#         result_table.loc[row_index, 'Model Name'] = est_name
#
#         cv_results = cross_validate(est,
#                                     X,
#                                     y,
#                                     cv=cv,
#                                     n_jobs=-1)
#
#         result_table.loc[row_index, 'Test accuracy'] = cv_results['test_score'].mean()
#         result_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
#         result_table.loc[row_index, 'Fit Time'] = cv_results['fit_time'].mean()
#
#         row_index += 1
#
#     result_table.sort_values(by=['Test accuracy'], ascending=False, inplace=True)
#
#     return result_table
#
#
#
#
#
#
#
# lr = LogisticRegression()
# dt = DecisionTreeClassifier(random_state=1)
# rf = RandomForestClassifier(random_state=1)
# svc = make_pipeline(StandardScaler(), SVC(probability=True))
# knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
#
# estimators = [lr,
#               dt,
#               rf,
#               svc,
#               knn, ]
#
# labels = ['Log Regression',
#           'Decision Tree',
#           'Random Forest',
#           'SVC',
#           'KNN', ]
#
# results = test_estimators(X_train, y_train, estimators, labels, cv=10)
# results.style.background_gradient(cmap='Blues')
#
# rf_params = {'min_samples_leaf': pd.np.arange(20, 50, 5),
#              'min_samples_split': pd.np.arange(20, 50, 5),
#              'max_depth': pd.np.arange(3, 6),
#              'min_weight_fraction_leaf': pd.np.arange(0, 0.4, 0.1),
#              'criterion': ['gini', 'entropy']}
#
# grid = GridSearchCV(rf, rf_params, scoring='average_precision', cv=10, n_jobs=-1)
#
# grid.fit(X_train, y_train)
#
# print ('the best choice')
# print(grid.best_estimator_)
# print(grid.best_params_)
# print(grid.best_score_)

#rf = RandomForestClassifier(**grid.best_params_)