import csv
import json
import os
import glob
import pandas as pd


path = './official-matrix-results/'
save_results = './official-matrix.csv'

runs = []

fieldnames = []

for filename in glob.glob(os.path.join(path, '*.json')):
    with open(filename, encoding='utf-8', mode='r') as f:
        data = json.load(f)
        run = pd.json_normalize(data).to_dict()

        for prop in run:
            run[prop] = run[prop][0]

        run['file'] = filename

        runs.append(run)

        for fieldname in run.keys():
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)

with open(save_results, 'w', newline='') as f:
    writer = csv.DictWriter(
        f, fieldnames=fieldnames)

    writer.writeheader()

    for run in runs:
        writer.writerow(run)
