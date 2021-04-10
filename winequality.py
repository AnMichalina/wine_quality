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




df.loc[df['fixed_acidity']<= 4.60, 'fixed_acidity'] =0
df.loc[(df['fixed_acidity'] > 4.60) & (df['fixed_acidity']<=7.10),'fixed_acidity'] =1
df.loc[(df['fixed_acidity'] > 7.10) & (df['fixed_acidity']<=7.90), 'fixed_acidity'] =2
df.loc[(df['fixed_acidity'] > 7.90) & (df['fixed_acidity'] <= 9.20), 'fixed_acidity'] =3
df.loc[df['fixed_acidity'] > 9.20, 'fixed_acidity'] =4

df.loc[df['volatile_acidity']<= 0.12, 'volatile_acidity'] =0
df.loc[(df['volatile_acidity'] > 0.12) & (df['volatile_acidity']<=0.39),'volatile_acidity'] =1
df.loc[(df['volatile_acidity'] > 0.39) & (df['volatile_acidity']<=0.52), 'volatile_acidity'] =2
df.loc[(df['volatile_acidity'] > 0.52) & (df['volatile_acidity'] <= 0.64), 'volatile_acidity'] =3
df.loc[df['volatile_acidity'] > 0.64, 'volatile_acidity'] =4

df.loc[df['citric_acid']<= 0.00, 'citric_acid'] =0
df.loc[(df['citric_acid'] > 0.00) & (df['citric_acid']<=0.09),'citric_acid'] =1
df.loc[(df['citric_acid'] > 0.09) & (df['citric_acid']<=0.26), 'citric_acid'] =2
df.loc[(df['citric_acid'] > 0.26) & (df['citric_acid'] <= 0.42), 'citric_acid'] =3
df.loc[df['citric_acid'] > 0.42, 'citric_acid'] =4

df.loc[df['residual_sugar']<= 0.90, 'residual_sugar'] =0
df.loc[(df['residual_sugar'] > 0.90) & (df['residual_sugar']<=1.90),'residual_sugar'] =1
df.loc[(df['residual_sugar'] > 1.90) & (df['residual_sugar']<=2.20), 'residual_sugar'] =2
df.loc[(df['citric_acid'] > 2.20) & (df['citric_acid'] <= 2.60), 'residual_sugar'] =3
df.loc[df['residual_sugar'] > 2.26, 'residual_sugar'] =4

df.loc[df['chlorides']<= 0.01, 'chlorides'] =0
df.loc[(df['chlorides'] > 0.01) & (df['chlorides']<=0.07),'chlorides'] =1
df.loc[(df['chlorides'] > 0.07) & (df['chlorides']<=0.08), 'chlorides'] =2
df.loc[(df['chlorides'] > 0.08) & (df['chlorides'] <= 0.09), 'chlorides'] =3
df.loc[df['chlorides'] > 0.09, 'chlorides'] =4

df.loc[df['free_sulfur_dioxide']<= 1.00, 'free_sulfur_dioxide'] =0
df.loc[(df['free_sulfur_dioxide'] > 1.00) & (df['free_sulfur_dioxide']<=7.00),'free_sulfur_dioxide'] =1
df.loc[(df['free_sulfur_dioxide'] > 7.00) & (df['free_sulfur_dioxide']<=14.00), 'free_sulfur_dioxide'] =2
df.loc[(df['free_sulfur_dioxide'] > 14.00) & (df['free_sulfur_dioxide'] <= 21.00), 'free_sulfur_dioxide'] =3
df.loc[df['free_sulfur_dioxide'] > 21.00, 'free_sulfur_dioxide'] =4

df.loc[df['total_sulfur_dioxide']<= 6.00, 'total_sulfur_dioxide'] =0
df.loc[(df['total_sulfur_dioxide'] > 6.00) & (df['total_sulfur_dioxide']<=22.00),'total_sulfur_dioxide'] =1
df.loc[(df['total_sulfur_dioxide'] > 22.00) & (df['total_sulfur_dioxide']<=38.00), 'total_sulfur_dioxide'] =2
df.loc[(df['total_sulfur_dioxide'] > 38.00) & (df['total_sulfur_dioxide'] <= 62.00), 'total_sulfur_dioxide'] =3
df.loc[df['total_sulfur_dioxide'] > 62.00, 'total_sulfur_dioxide'] =4

df.loc[df['density']<= 0.99, 'density'] =0
df.loc[(df['density'] > 0.99) & (df['density']<=1.00),'density'] =1

df.loc[df['pH']<= 2.74, 'pH'] =0
df.loc[(df['pH'] > 2.74) & (df['pH']<=3.21),'pH'] =1
df.loc[(df['pH'] > 3.21) & (df['pH']<=3.31), 'pH'] =2
df.loc[(df['pH'] > 3.31) & (df['pH'] <= 3.40), 'pH'] =3
df.loc[df['pH'] > 3.40, 'pH'] =4

df.loc[df['sulphates']<= 0.33, 'sulphates'] =0
df.loc[(df['sulphates'] > 0.33) & (df['sulphates']<=0.55),'sulphates'] =1
df.loc[(df['sulphates'] > 0.55) & (df['sulphates']<=0.62), 'sulphates'] =2
df.loc[(df['sulphates'] > 0.62) & (df['sulphates'] <= 0.73), 'sulphates'] =3
df.loc[df['sulphates'] > 0.73, 'sulphates'] =4


df.loc[df['alcohol']<= 8.40, 'alcohol'] =0
df.loc[(df['alcohol'] > 8.40) & (df['alcohol']<=9.50),'alcohol'] =1
df.loc[(df['alcohol'] > 9.50) & (df['alcohol']<=10.20), 'alcohol'] =2
df.loc[(df['alcohol'] > 10.20) & (df['alcohol'] <= 11.10), 'alcohol'] =3
df.loc[df['alcohol'] > 11.10, 'alcohol'] =4
df.head()

df.loc[df['quality']<= 7.00, 'quality'] =0
df.loc[(df['quality'] > 7.00) & (df['quality']<=10.00),'quality'] =1

print (round(df.describe(),2))

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

X.info()
print (X.isnull().values.any())
print (X.isna().sum())

y = df['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


cols = X_train.columns
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

rf = RandomForestClassifier(max_depth=3, min_samples_leaf=20, min_samples_split=20, random_state= 1, criterion='gini')
rf.fit(X_train, y_train)
y_pred1 = rf.predict(X_test)

print('Model accuracy score of KNN: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
print('Model accuracy score of RF: {0:0.4f}'. format(accuracy_score(y_test, y_pred1)))


# def test_estimators(X, y, estimators, labels, cv):
#     '''
#     A function for testing multiple estimators.
# #     It takes: full train data and target, list of estimators,
# #               list of labels or names of estimators,
# #               cross validation splitting strategy;
# #     And it returns: a DataFrame of table with results of tests
# #     '''
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
# lr = LogisticRegression()
# dt = DecisionTreeClassifier(random_state=1)
# rf = RandomForestClassifier(random_state=1)
# svc = make_pipeline(MinMaxScaler(), SVC(probability=True))
# knn = make_pipeline(MinMaxScaler(), KNeighborsClassifier())
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
# rf_params = {'min_samples_leaf': pd.np.arange(20),
#              'min_samples_split': pd.np.arange(20),
#              'max_depth': pd.np.arange(3),
#              'min_weight_fraction_leaf': pd.np.arange(0),
#              'criterion': ['gini']}
#
# grid = GridSearchCV(rf, rf_params, scoring='average_precision', cv=10, n_jobs=-1)
#
# grid.fit(X_train, y_train)
#
# print ('the best choice')
# print(grid.best_estimator_)
# print(grid.best_params_)
# print(grid.best_score_)
#
# rf = RandomForestClassifier(**grid.best_params_)
