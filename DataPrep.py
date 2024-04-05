from azureml.core import Run
new_run = Run.get_context()
ws = new_run.experiment.workspace

def data_prep():
    input_ds = ws.datasets.get("gfdhnbv").to_pandas_dataframe()
    input_ds = input_ds.drop(['ID', "Gender"], axis=1)
    input_ds = input_ds.iloc[:50, :]
    return input_ds
