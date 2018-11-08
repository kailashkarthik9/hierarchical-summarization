import numpy as np
import pandas as pd

expected_output_path = "F:/Text Summarization/Neural Network Dataset/Output/test/"
actual_output_path = "F:/Text Summarization/Neural Network Dataset/Result/"

chunk_size = 10

for x in range(1, 1091):
    print("evaluating file " + str(x))
    df_expected = pd.read_csv(expected_output_path + str(x) + '.csv', header=None)
    df_expected = df_expected.append(pd.DataFrame(np.zeros((chunk_size - (df_expected.shape[0] % chunk_size), 1))),
                                     ignore_index=True)
    df_actual = pd.read_csv(actual_output_path + str(x) + '.csv', header=None)
    output_size = df_expected.shape[0]
    tp = 0
    tn = 0
    fp = 0
    fn = 0
