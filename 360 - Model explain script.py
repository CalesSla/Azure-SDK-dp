from azureml.core import Run

new_run = Run.get_context()
ws = new_run.experiment.workspace

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str)
args = parser.parse_args()

import pandas as pd

df = new_run.input_datasets['raw_data'].to_pandas_dataframe()
dataPrep = pd.get_dummies(df, drop_first=True)

X = dataPrep.iloc[:, :-1]
Y = dataPrep.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state=1234, stratify=Y)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)

trained_model = rfc.fit(X_train, Y_train)

Y_predict = rfc.predict(X_test)
Y_prob = rfc.predict_proba(X_test)[:, 1]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)
score = rfc.score(X_test, Y_test)

new_run.log("accuracu", score)

from interpret.ext.blackbox import TabularExplainer

features = list(X.columns)
classes = ["notGreater", "Greater"]

tab_explainer = TabularExplainer(trained_model,
                                 X_train,
                                 features=features,
                                 classes=classes)

explanation = tab_explainer.explain_global(X_train)

# Upload the explanations to the workspace
from azureml.interpret import ExplanationClient
explain_client = ExplanationClient.from_run(new_run)

explain_client.upload_model_explanation(explanation,
                                        comment="My First Explanations")

new_run.complete()