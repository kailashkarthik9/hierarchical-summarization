import keras
import numpy as np
import pandas as pd

chunk_size = 10

model = keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(keras.layers.Dense(3020, activation='relu'))
# Add another:
model.add(keras.layers.Dense(1515, activation='relu', bias_regularizer=keras.regularizers.l1(0.01)))
# Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(
    optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
    loss=keras.losses.categorical_crossentropy,
    metrics=[keras.metrics.categorical_accuracy])

input_data_path = "F:/Text Summarization/Neural Network Dataset/4. Caption and Headline/training/"
input_labels_path = "F:/Text Summarization/Neural Network Dataset/Output/training/"

for x in range(1, 83565):
    print("training with file " + str(x))
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
    model.fit(feed_csv.values, feed_csv_output.values, epochs=50)

model.save('batch10.h5')
