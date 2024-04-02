from azureml.core import Workspace, Datastore, Dataset, Experiment, Run
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

ws = Workspace.from_config("./config")
az_store = Datastore.get(ws, "azure_sdk_blob01")
az_dataset = Dataset.get_by_name(ws, "Loan Data using SDK")
az_default_store = ws.get_default_datastore()

# Run experiment using start_logging method
new_run = Run.get_context()

df = az_dataset.to_pandas_dataframe()

total_observations = len(df)
nulldf = df.isnull().sum()

new_df = df[["age", "workclass", "education", "occupation"]]
new_df.to_csv("./outputs/loan_trunc.csv", index=False)

new_run.log("Total Observations", total_observations)

for columns in df.columns:
    new_run.log(columns, nulldf[columns])

new_run.complete()
