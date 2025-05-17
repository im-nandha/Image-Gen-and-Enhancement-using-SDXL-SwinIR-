import yaml
import subprocess

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("Starting training...")
subprocess.call(["python", "train_sdxl.py"])

print("Running inference...")
subprocess.call(["python", "inference.py"])

print("Evaluating...")
subprocess.call(["python", "evaluate.py"])
