import os
import yaml

import mlflow
from mlflow.tracking import MlflowClient

mlruns_path = "./mlruns"  # or absolute path if needed
print("here")


def check_mlflow_runs(mlruns_path):
    for exp_id in os.listdir(mlruns_path):
        exp_path = os.path.join(mlruns_path, exp_id)
        if not os.path.isdir(exp_path):
            continue

        for run_id in os.listdir(exp_path):
            if run_id == 'metal.yaml':
                print("here")
                continue
            run_path = os.path.join(exp_path, run_id)
            meta_path = os.path.join(run_path, "meta.yaml")

            try:
                with open(meta_path, "r") as f:
                    meta = yaml.safe_load(f)
                    if not isinstance(meta, dict):
                        print(f"Corrupt or empty meta.yaml: {run_path}")
            except Exception as e:
                pass


experiment_id = "824047249790940545"  # your experiment ID
metric_name = "Train total loss"
print("experiment_id", experiment_id)
print("metric_name", metric_name)
client = MlflowClient()

# List all runs in the experiment
runs = client.search_runs(experiment_ids=[experiment_id])
print("Starting to check runs...")
for run in runs:
    run_id = run.info.run_id
    print("Run ID:", run_id)
    try:
        # Try loading the metric history
        metric_history = client.get_metric_history(run_id, metric_name)
        for m in metric_history:
            if not hasattr(m, "step") or not hasattr(m, "value"):
                print(f"⚠️ Malformed metric in run: {run_id}")
                break
    except Exception as e:
        print(f"❌ Error loading run {run_id}: {e}")
