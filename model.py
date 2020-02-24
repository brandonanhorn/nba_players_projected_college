import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn import tree

df = pd.read_csv('data/bio_stats.csv')

target = 'college'
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

mapper = DataFrameMapper([
    ('player_height', LabelBinarizer()),
    ('country', LabelBinarizer()),
    ('draft_year', LabelBinarizer()),
    ('draft_round', LabelBinarizer()),
    ('draft_number', LabelBinarizer())],df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

model = DecisionTreeClassifier()
model.fit(Z_train, y_train)
model.score(Z_train, y_train)
model.score(Z_test, y_test)

pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
pickle.dump(pipe, open('pipe.pkl', 'wb'))

X_train.sample().to_dict(orient = 'list')

new = pd.DataFrame({
 'player_height': ['6-8'],
 'player_weight': [250],
 'country': ['USA'],
 'draft_year': ['2001'],
 'draft_round': ['1'],
 'draft_number': ['30']})


type(pipe.predict(new)[0])

prediction = (pipe.predict(new)[0])
prediction
