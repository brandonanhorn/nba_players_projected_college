import pandas as pd
import seaborn as sns
import catboost
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier
from sklearn import preprocessing

df = pd.read_csv('data/bio_stats.csv')

df['college'].value_counts()

target = 'college'
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

mapper = DataFrameMapper([
    ('player_height', LabelBinarizer()),
    ('player_weight', LabelBinarizer()),
    ('country', LabelBinarizer()),
    ('draft_year', LabelBinarizer()),
    ('draft_round', LabelBinarizer()),
    ('draft_number', LabelBinarizer())],df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

model = LogisticRegression(max_iter=500).fit(Z_train, y_train)
model.score(Z_train, y_train)
model.score(Z_test, y_test)

model = RandomForestClassifier().fit(Z_train, y_train)
model.score(Z_train, y_train)
model.score(Z_test, y_test)

model = DecisionTreeClassifier().fit(Z_train, y_train)
model.score(Z_train, y_train)
model.score(Z_test, y_test)

model =CatBoostClassifier(iterations=100).fit(Z_train, y_train)
model.score(Z_train, y_train)
model.score(Z_test,y_test)
