import numpy as np
import pandas as pd
from keras.models import load_model

chunk_size = 10
model = load_model('batch10.h5')

input_data_path = "F:/Text Summarization/Neural Network Dataset/4. Caption and Headline/test/"
input_labels_path = "F:/Text Summarization/Neural Network Dataset/Output/test/"
output_path = "F:/Text Summarization/Neural Network Dataset/Result/"

for x in range(1, 1091):
    print("testing with file " + str(x))
    big_csv = pd.DataFrame()
    big_csv_output = pd.DataFrame()
    df = pd.read_csv(input_data_path + str(x) + '.csv', header=None)
    if (df.shape[0] == 302 and df.shape[1] == 1):
        df = df.T
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
    model_output = model.predict(feed_csv.values)
    output_builder = pd.DataFrame()
    for row in model_output:
        output_builder = output_builder.append(pd.DataFrame(row))
    np.savetxt(output_path + str(x) + ".csv", output_builder, delimiter=",")
