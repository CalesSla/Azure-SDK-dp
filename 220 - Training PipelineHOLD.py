from azureml.core import Run
import argparse

new_run = Run.get_context()

ws = new_run.experiment.workspace

parser = argparse.ArgumentParser()
parser.add_argument("--datafolder", type=str)
args = parser.parse_args()

import os
import pandas as pd
path = os.path.join(args.datafolder, "defaults_prep.csv")
dataPrep = pd.read_csv(path)

Y = dataPrep[["Default Next Month_Yes"]]
X = dataPrep.drop(["Default Next Month_Yes"], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train, Y_train)

Y_predict = lr.predict(X_test)
Y_prob = lr.predict_proba(X_test)[:, 1]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)
score = lr.score(X_test, Y_test)

cm_dict = {
                "schema_type": "confusion_matrix",
                "schema_version": "1.0.0",
                "data": {
                    "class_labels": ["N", "Y"],
                    "matrix": cm.tolist()
                }
            }

new_run.log("TotalObservations", len(dataPrep))
new_run.log_confusion_matrix("ConfusionMatrix", cm_dict)
new_run.log("Score", score)

X_test = X_test.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)

Y_prob_df = pd.DataFrame(Y_prob, columns=["Scored Probabilities"])
Y_predict_df = pd.DataFrame(Y_predict, columns=["Scored Label"])

scored_dataset = pd.concat([X_test, Y_test, Y_predict_df, Y_prob_df], axis=1)

scored_dataset.to_csv("./outputs/defaults_scored.csv", index=False)

new_run.complete()