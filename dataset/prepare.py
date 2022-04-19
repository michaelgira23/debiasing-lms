"""
Combine WinoBias and Crows-Pairs datasets and split into training, cross-validation, and testing sets.
"""

import csv
import numpy as np
import re
from sklearn.model_selection import train_test_split

train_path = 'train.txt'
train_ratio = 0.80

cross_validation_path = 'cross_validation.txt'
cross_validation_ratio = 0.10

test_path = 'test.txt'
test_ratio = 0.10

combined_path = 'combined.txt'

winobias_paths = [
    './corefBias/WinoBias/wino/data/anti_stereotyped_type1.txt.dev',
    './corefBias/WinoBias/wino/data/anti_stereotyped_type2.txt.dev',
    './corefBias/WinoBias/wino/data/pro_stereotyped_type1.txt.dev',
    './corefBias/WinoBias/wino/data/pro_stereotyped_type2.txt.dev',
]

crowspairs_paths = [
    './crows-pairs/data/crows_pairs_anonymized.csv',
]

# Combine all dataset files into a single combined.txt
with open('combined.txt', 'w', newline='\n') as combinedfile:
    for path in winobias_paths:
        with open(path, 'r', newline='\n') as f:
            data = f.read()

            # Remove brackets
            data = data.replace('[', '')
            data = data.replace(']', '')

            # Remove numbering
            data = re.sub(r'\d+ ', '', data)

            combinedfile.write(data)

    for path in crowspairs_paths:
        with open(path, 'r', newline='\n') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Remove newlines in the middle of each sentence
                combinedfile.writelines(
                    [row['sent_more'].replace('\n', ''), '\n'])
                combinedfile.writelines(
                    [row['sent_less'].replace('\n', ''), '\n'])

# Split into training, cross-validation, and testing sets
with open('combined.txt', 'r', newline='\n') as combinedfile:
    # Split combined into
    nonempty_lines = [line.strip('\n')
                      for line in combinedfile if line != '\n']
    line_count = len(nonempty_lines)

    # Pair corresponding sentences together
    corresponding_sentences = np.array(
        nonempty_lines).reshape(int(line_count / 2), 2)
    labels = np.zeros(int(line_count / 2))

    X_train, X_remain, y_train, y_remain = train_test_split(
        corresponding_sentences, labels, test_size=1 - train_ratio, random_state=42)

    X_cv, X_test, y_cv, y_test = train_test_split(
        X_remain, X_remain, test_size=test_ratio/(test_ratio + cross_validation_ratio), random_state=42)

    with open(train_path, 'w', newline='\n') as outfile:
        outfile.writelines('\n'.join(X_train.flatten().tolist()))

    with open(cross_validation_path, 'w', newline='\n') as outfile:
        outfile.writelines('\n'.join(X_cv.flatten().tolist()))

    with open(test_path, 'w', newline='\n') as outfile:
        outfile.writelines('\n'.join(X_test.flatten().tolist()))
