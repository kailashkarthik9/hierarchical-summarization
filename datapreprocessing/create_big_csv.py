import numpy as np
import pandas as pd

test_path = r'''F:\Text Summarization\Neural Network Dataset\4. Caption and Headline\test\\'''
test_output_path = r'''F:\Text Summarization\Neural Network Dataset\Output\test\\'''
training_path = r'''F:\Text Summarization\Neural Network Dataset\4. Caption and Headline\training\\'''
training_output_path = r'''F:\Text Summarization\Neural Network Dataset\Output\training\\'''
chunk_size = 10

# big_csv = pd.DataFrame()
# big_csv_output = pd.DataFrame()
# for x in range(1, 1091):
#     print('test' + str(x))
#     df = pd.read_csv(test_path + str(x) + '.csv', header=None)
#     df_output = pd.read_csv(test_output_path + str(x) + '.csv', header=None)
#     big_csv = big_csv.append(df.append(pd.DataFrame(np.zeros((chunk_size - (df.shape[0] % chunk_size), 302))), ignore_index = True), ignore_index = True)
#     big_csv_output = big_csv_output.append(df_output.append(pd.DataFrame(np.zeros((chunk_size - (df_output.shape[0] % chunk_size), 1))), ignore_index = True), ignore_index = True)
# big_csv.to_csv('F:\\test_text_chunk_' + str(chunk_size) + '.csv', index = False, header = False)
# big_csv_output.to_csv('F:\\test_text_chunk_' + str(chunk_size) + '_output.csv', index = False, header = False)
#
# for i in range(0, 1):
#     big_csv = pd.DataFrame()
#     big_csv_output = pd.DataFrame()
#     for x in range(i*1000 + 1, (i+1)*1000 + 1):
#         print(str(x))
#         df = pd.read_csv(training_path + str(x) + '.csv', header=None)
#         df_output = pd.read_csv(training_output_path + str(x) + '.csv', header=None)
#         big_csv = big_csv.append(df.append(pd.DataFrame(np.zeros((chunk_size - (df.shape[0] % chunk_size), 302))), ignore_index = True), ignore_index = True)
#         big_csv_output = big_csv_output.append(df_output.append(pd.DataFrame(np.zeros((chunk_size - (df_output.shape[0] % chunk_size), 1))), ignore_index = True), ignore_index = True)
#     big_csv.to_csv('F:\\training_text_chunk_' + str(chunk_size) + '.csv', index = False, header = False)
#     big_csv_output.to_csv('F:\\training_text_chunk_' + str(chunk_size) + '_output.csv', index = False, header = False)

for i in range(8, 82):
    big_csv = pd.DataFrame()
    big_csv_output = pd.DataFrame()
    for x in range(i * 1000 + 1, (i + 1) * 1000 + 1):
        print(str(x))
        df = pd.read_csv(training_path + str(x) + '.csv', header=None)
        df_output = pd.read_csv(training_output_path + str(x) + '.csv', header=None)
        big_csv = big_csv.append(
            df.append(pd.DataFrame(np.zeros((chunk_size - (df.shape[0] % chunk_size), 301))), ignore_index=True),
            ignore_index=True)
        big_csv_output = big_csv_output.append(
            df_output.append(pd.DataFrame(np.zeros((chunk_size - (df_output.shape[0] % chunk_size), 1))),
                             ignore_index=True), ignore_index=True)
    pd.read_csv('F:\\training_text_chunk_' + str(chunk_size) + '.csv', header=None).append(big_csv).to_csv(
        'F:\\training_text_chunk_' + str(chunk_size) + '.csv', index=False, header=False)
    pd.read_csv('F:\\training_text_chunk_' + str(chunk_size) + '_output.csv', header=None).append(
        big_csv_output).to_csv(
        'F:\\training_text_chunk_' + str(chunk_size) + '_output.csv', index=False, header=False)

chunk_size = 20

big_csv = pd.DataFrame()
big_csv_output = pd.DataFrame()
for x in range(1, 1091):
    print('test' + str(x))
    df = pd.read_csv(test_path + str(x) + '.csv', header=None)
    df_output = pd.read_csv(test_output_path + str(x) + '.csv', header=None)
    big_csv = big_csv.append(
        df.append(pd.DataFrame(np.zeros((chunk_size - (df.shape[0] % chunk_size), 302))), ignore_index=True),
        ignore_index=True)
    big_csv_output = big_csv_output.append(
        df_output.append(pd.DataFrame(np.zeros((chunk_size - (df_output.shape[0] % chunk_size), 1))),
                         ignore_index=True), ignore_index=True)
big_csv.to_csv('F:\\test_text_chunk_' + str(chunk_size) + '.csv', index=False, header=False)
big_csv_output.to_csv('F:\\test_text_chunk_' + str(chunk_size) + '_output.csv', index=False, header=False)

for i in range(0, 1):
    big_csv = pd.DataFrame()
    big_csv_output = pd.DataFrame()
    for x in range(i * 1000 + 1, (i + 1) * 1000 + 1):
        print(str(x))
        df = pd.read_csv(training_path + str(x) + '.csv', header=None)
        df_output = pd.read_csv(training_output_path + str(x) + '.csv', header=None)
        big_csv = big_csv.append(
            df.append(pd.DataFrame(np.zeros((chunk_size - (df.shape[0] % chunk_size), 302))), ignore_index=True),
            ignore_index=True)
        big_csv_output = big_csv_output.append(
            df_output.append(pd.DataFrame(np.zeros((chunk_size - (df_output.shape[0] % chunk_size), 1))),
                             ignore_index=True), ignore_index=True)
    big_csv.to_csv('F:\\training_text_chunk_' + str(chunk_size) + '.csv', index=False, header=False)
    big_csv_output.to_csv('F:\\training_text_chunk_' + str(chunk_size) + '_output.csv', index=False, header=False)

for i in range(1, 82):
    big_csv = pd.DataFrame()
    big_csv_output = pd.DataFrame()
    for x in range(i * 1000 + 1, (i + 1) * 1000 + 1):
        print(str(x))
        df = pd.read_csv(training_path + str(x) + '.csv', header=None)
        df_output = pd.read_csv(training_output_path + str(x) + '.csv', header=None)
        big_csv = big_csv.append(
            df.append(pd.DataFrame(np.zeros((chunk_size - (df.shape[0] % chunk_size), 301))), ignore_index=True),
            ignore_index=True)
        big_csv_output = big_csv_output.append(
            df_output.append(pd.DataFrame(np.zeros((chunk_size - (df_output.shape[0] % chunk_size), 1))),
                             ignore_index=True), ignore_index=True)
    pd.read_csv('F:\\training_text_chunk_' + str(chunk_size) + '.csv', header=None).append(big_csv).to_csv(
        'F:\\training_text_chunk_' + str(chunk_size) + '.csv', index=False, header=False)
    pd.read_csv('F:\\training_text_chunk_' + str(chunk_size) + '_output.csv', header=None).append(
        big_csv_output).to_csv(
        'F:\\training_text_chunk_' + str(chunk_size) + '_output.csv', index=False, header=False)

chunk_size = 30

big_csv = pd.DataFrame()
big_csv_output = pd.DataFrame()
for x in range(1, 1091):
    print('test' + str(x))
    df = pd.read_csv(test_path + str(x) + '.csv', header=None)
    df_output = pd.read_csv(test_output_path + str(x) + '.csv', header=None)
    big_csv = big_csv.append(
        df.append(pd.DataFrame(np.zeros((chunk_size - (df.shape[0] % chunk_size), 302))), ignore_index=True),
        ignore_index=True)
    big_csv_output = big_csv_output.append(
        df_output.append(pd.DataFrame(np.zeros((chunk_size - (df_output.shape[0] % chunk_size), 1))),
                         ignore_index=True), ignore_index=True)
big_csv.to_csv('F:\\test_text_chunk_' + str(chunk_size) + '.csv', index=False, header=False)
big_csv_output.to_csv('F:\\test_text_chunk_' + str(chunk_size) + '_output.csv', index=False, header=False)

for i in range(0, 1):
    big_csv = pd.DataFrame()
    big_csv_output = pd.DataFrame()
    for x in range(i * 1000 + 1, (i + 1) * 1000 + 1):
        print(str(x))
        df = pd.read_csv(training_path + str(x) + '.csv', header=None)
        df_output = pd.read_csv(training_output_path + str(x) + '.csv', header=None)
        big_csv = big_csv.append(
            df.append(pd.DataFrame(np.zeros((chunk_size - (df.shape[0] % chunk_size), 302))), ignore_index=True),
            ignore_index=True)
        big_csv_output = big_csv_output.append(
            df_output.append(pd.DataFrame(np.zeros((chunk_size - (df_output.shape[0] % chunk_size), 1))),
                             ignore_index=True), ignore_index=True)
    big_csv.to_csv('F:\\training_text_chunk_' + str(chunk_size) + '.csv', index=False, header=False)
    big_csv_output.to_csv('F:\\training_text_chunk_' + str(chunk_size) + '_output.csv', index=False, header=False)

for i in range(1, 82):
    big_csv = pd.DataFrame()
    big_csv_output = pd.DataFrame()
    for x in range(i * 1000 + 1, (i + 1) * 1000 + 1):
        print(str(x))
        df = pd.read_csv(training_path + str(x) + '.csv', header=None)
        df_output = pd.read_csv(training_output_path + str(x) + '.csv', header=None)
        big_csv = big_csv.append(
            df.append(pd.DataFrame(np.zeros((chunk_size - (df.shape[0] % chunk_size), 301))), ignore_index=True),
            ignore_index=True)
        big_csv_output = big_csv_output.append(
            df_output.append(pd.DataFrame(np.zeros((chunk_size - (df_output.shape[0] % chunk_size), 1))),
                             ignore_index=True), ignore_index=True)
    pd.read_csv('F:\\training_text_chunk_' + str(chunk_size) + '.csv', header=None).append(big_csv).to_csv(
        'F:\\training_text_chunk_' + str(chunk_size) + '.csv', index=False, header=False)
    pd.read_csv('F:\\training_text_chunk_' + str(chunk_size) + '_output.csv', header=None).append(
        big_csv_output).to_csv(
        'F:\\training_text_chunk_' + str(chunk_size) + '_output.csv', index=False, header=False)
