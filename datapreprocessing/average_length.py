from fractions import gcd
from functools import reduce

import pandas as pd

test_path = r'''F:\Text Summarization\Neural Network Dataset\3. Caption\test\\'''
training_path = r'''F:\Text Summarization\Neural Network Dataset\3. Caption\training\\'''

total_count = 0
max_count = 0
min_count = 1000
file_counts = []

for x in range(1, 1091):
    print(str(x))
    df = pd.read_csv(test_path + str(x) + '.csv', header=None)
    file_count = df.shape[0]
    file_counts.append(file_count)
    total_count += file_count
    if file_count > max_count:
        max_count = file_count
    elif file_count < min_count:
        min_count = file_count

print(max_count)
print(min_count)
print(total_count)
print(total_count / 1090)
print(reduce(gcd, file_counts))
