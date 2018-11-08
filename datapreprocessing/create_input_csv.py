import numpy as np
import pandas as pd

input_data_path = "F:/Text Summarization/Neural Network Dataset/4. Caption and Headline/training/"
input_labels_path = "F:/Text Summarization/Neural Network Dataset/Output/training/"
output_data_path = "F:/Text Summarization/Neural Network Dataset/batch10/data/training/"
output_labels_path = "F:/Text Summarization/Neural Network Dataset/batch10/labels/training/"

chunk_size = 10

for x in range(1, 83565):
    print("test" + str(x))
    big_csv = pd.DataFrame()
    big_csv_output = pd.DataFrame()
    df = pd.read_csv(input_data_path + str(x) + '.csv', header=None)
    df_output = pd.read_csv(input_labels_path + str(x) + '.csv', header=None)
    big_csv = df.append(pd.DataFrame(np.zeros((chunk_size - (df.shape[0] % chunk_size), 302))), ignore_index=True)
    big_csv_output = df_output.append(pd.DataFrame(np.zeros((chunk_size - (df_output.shape[0] % chunk_size), 1))),
                                      ignore_index=True)
    n = big_csv.shape[0]
    feed_csv = pd.DataFrame()
    feed_csv_output = pd.DataFrame()
    for i in range(0, n):
        if i % 10 == 0:
            append_df = big_csv.iloc[i].to_frame().T
            append_df_out = big_csv_output.iloc[i].to_frame().T
            for j in range(1, 10):
                append_df = np.c_[append_df, big_csv.iloc[i + j].to_frame().T]
                append_df_out = np.c_[append_df_out, big_csv_output.iloc[i + j].to_frame().T]
            feed_csv = feed_csv.append(pd.DataFrame(append_df))
            feed_csv_output = feed_csv_output.append(pd.DataFrame(append_df_out))
    feed_csv.to_csv(output_data_path + str(x) + '.csv', index=False, header=False)
    feed_csv_output.to_csv(output_labels_path + str(x) + '.csv', index=False, header=False)
