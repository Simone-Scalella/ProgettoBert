import json
import csv
import io
import os
import pandas as pd

path_file = os.path.abspath(os.getcwd())
print(path_file)
os.chdir(path_file)

df = pd.read_json('data.jsonl', lines=True)
df.to_csv('data.csv')