from azureml.core import Run

new_run = Run.get_context()
ws = new_run.experiment.workspace

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int)
parser.add_argument("--min_samples_leaf", type=int)
parser.add_argument("--input-data", type=str)
args = parser.parse_args()

ne = args.n_estimators
msl = args.min_samples_leaf

import pandas as pd

df = new_run.input_datasets['raw_data'].to_pandas_dataframe()

dataPrep = df.drop(["ID"], axis=1)
dataPrep = dataPrep.dropna()
dataPrep = pd.get_dummies(dataPrep, drop_first=True)
Y = dataPrep[["Default Next Month_Yes"]]
X = dataPrep.drop(["Default Next Month_Yes"], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)

from sklearn.ensemble import RandomForestClassifier as RFC

rfc = RFC(n_estimators=ne, min_samples_leaf=msl)
rfc.fit(X_train, Y_train)
Y_predict = rfc.predict(X_test)

Y_prob = rfc.predict_proba(X_test)[:, 1]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)
score = rfc.score(X_test, Y_test)

new_run.log("accuracy", score)
new_run.complete()