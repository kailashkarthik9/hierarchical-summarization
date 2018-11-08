output_path = "F:/Text Summarization/Neural Network Dataset/Output/test/"
combined_path = "F:/Text Summarization/Neural Network Dataset/4. Caption and Headline/test/"
headline_path = "F:/Text Summarization/Neural Network Dataset/2. Headline/test/"
caption_path = "F:/Text Summarization/Neural Network Dataset/3. Caption/test/"
baseline_path = "F:/Text Summarization/Neural Network Dataset/1. Baseline/test/"

# for x in range(1, 1091):
#     df_caption = pd.read_csv(output_path + str(x) + '.csv', header=None)
#     df_baseline = pd.read_csv(baseline_path + str(x) + '.csv', header=None)
#     print(str(x) + ' : ' + str(df_baseline.shape[0]) + ' : ' + str(df_caption.shape[0]))
#     if (df_caption.shape[0] != df_baseline.shape[0]):
#         print('check the code again! for file ' + str(x))

# for x in range(1, 1091):
#     df_caption = pd.read_csv(caption_path + str(x) + '.csv', header=None)
#     df_baseline = pd.read_csv(baseline_path + str(x) + '.csv', header=None)
#     print(str(x) + ' : ' + str(df_baseline.shape[0]) + ' : ' + str(df_caption.shape[0]))
#     if (df_caption.shape[0] != df_baseline.shape[0]):
#         print('check the code again! for file ' + str(x))

# for x in range(1, 1091):
#     df_headline = pd.read_csv(headline_path + str(x) + '.csv', header=None)
#     df_baseline = pd.read_csv(baseline_path + str(x) + '.csv', header=None)
#     print(str(x) + ' : ' + str(df_baseline.shape[0]) + ' : ' + str(df_headline.shape[0]))
#     if (df_headline.shape[0] != df_baseline.shape[0]):
#         print('check the code again! for file ' + str(x))

# for x in range(1, 1091):
#     df_combined = pd.read_csv(combined_path + str(x) + '.csv', header=None)
#     df_baseline = pd.read_csv(baseline_path + str(x) + '.csv', header=None)
#     print(str(x) + ' : ' + str(df_baseline.shape[0]) + ' : ' + str(df_combined.shape[0]))
#     if (df_combined.shape[0] != df_baseline.shape[0]):
#         print('check the code again! for file ' + str(x))

# for x in range(1, 1091):
#     df_headline = pd.read_csv(headline_path + str(x) + '.csv', header=None)
#     df_headline2 = pd.read_csv(headline_path2 + str(x) + '.csv', header=None)
#     if not df_headline.equals(df_headline2):
#         print('nooooo')

# for x in range(1, 1091):
#     df_combined = pd.read_csv(combined_path + str(x) + '.csv', header=None)
#     df_combined2 = pd.read_csv(combined_path2 + str(x) + '.csv', header=None)
#     if not df_combined.equals(df_combined2):
#         print('nooooo')
