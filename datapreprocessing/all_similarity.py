import math

import numpy as np
from scipy import spatial

# text_path = "F:/Text Summarization/CNN Corpus/2_article text/csv/test/"
# caption_path = "F:/Text Summarization/CNN Corpus/4_image captions/csv/test/"
# summary_path = "F:/Text Summarization/CNN Corpus/5_highlights/csv/test/"
# headline_path = "F:/Text Summarization/CNN Corpus/3_title/csv/test/"
# out_caption_path = "F:/Text Summarization/Neural Network Dataset/3. Caption/test/"
# out_caption_data = np.array([])
# out_summary_path = "F:/Text Summarization/Neural Network Dataset/Output/test/"
# out_summary_data = np.array([])
# out_headline_path = "F:/Text Summarization/Neural Network Dataset/2. Headline/test/"
# out_headline_data = np.array([])
# out_combined_data = np.array([])
# out_combined_path = "F:/Text Summarization/Neural Network Dataset/4. Caption and Headline/test/"
#
# for x in range(1, 1091):
#     print("test" + str(x))
#     text_data = np.genfromtxt(text_path + str(x) + ".csv", dtype=float, delimiter=',')
#
#     article_text_lines_count = 1
#     if text_data.ndim == 2:
#         article_text_lines_count = text_data.shape[0]
#     # empty summary data
#     out_summary_data = np.zeros(article_text_lines_count)
#     # initialize caption data with extra column
#     if article_text_lines_count == 1:
#         out_caption_data = np.append(text_data, np.zeros(article_text_lines_count))
#     else:
#         out_caption_data = np.c_[text_data, np.zeros(np.size(text_data, 0))]
#     # initialize headline data with extra column
#     if article_text_lines_count == 1:
#         out_headline_data = np.append(text_data, np.zeros(article_text_lines_count))
#     else:
#         out_headline_data = np.c_[text_data, np.zeros(np.size(text_data, 0))]
#
#     caption_data = np.genfromtxt(caption_path + str(x) + ".csv", dtype=float, delimiter=',')
#     if caption_data.size > 0:
#         caption_lines_count = 1
#         if caption_data.ndim == 2:
#             caption_lines_count = caption_data.shape[0]
#         for i in range(0, caption_lines_count):
#             matched_index = -1;
#             matched_value = math.inf
#             for j in range(0, article_text_lines_count):
#                 dist = spatial.distance.cosine(caption_data[i], text_data[j])
#                 if dist < matched_value:
#                     matched_index = j
#                     matched_value = dist
#             if article_text_lines_count == 1:
#                 out_caption_data[-1] = 0.569
#             else:
#                 out_caption_data[matched_index, -1] = 0.569
#     np.savetxt(out_caption_path + str(x) + ".csv", out_caption_data, delimiter=",")
#
#     summary_data = np.genfromtxt(summary_path + str(x) + ".csv", dtype=float, delimiter=',')
#     if summary_data.size > 0:
#         summary_lines_count = 1
#         if summary_data.ndim == 2:
#             summary_lines_count = summary_data.shape[0]
#         for i in range(0, summary_lines_count):
#             matched_index = -1;
#             matched_value = math.inf
#             for j in range(0, article_text_lines_count):
#                 dist = spatial.distance.cosine(summary_data[i], text_data[j])
#                 if dist < matched_value:
#                     matched_index = j
#                     matched_value = dist
#             out_summary_data[matched_index] = 1
#     else:
#         print('no summary?')
#     np.savetxt(out_summary_path + str(x) + ".csv", out_summary_data, delimiter=",")
#
#     headline_data = np.genfromtxt(headline_path + str(x) + ".csv", dtype=float, delimiter=',')
#     if headline_data.size > 0:
#         headline_lines_count = 1
#         if headline_data.ndim == 2:
#             headline_lines_count = headline_data.shape[0]
#         for i in range(0, headline_lines_count):
#             matched_index = -1;
#             matched_value = math.inf
#             for j in range(0, article_text_lines_count):
#                 dist = spatial.distance.cosine(headline_data[i], text_data[j])
#                 if dist < matched_value:
#                     matched_index = j
#                     matched_value = dist
#             if article_text_lines_count == 1:
#                 out_headline_data[-1] = 0.569
#             else:
#                 out_headline_data[matched_index, -1] = 0.569
#     np.savetxt(out_headline_path + str(x) + ".csv", out_headline_data, delimiter=",")
#
#     only_caption_data = np.genfromtxt(out_caption_path + str(x) + ".csv", dtype=float, delimiter=',')
#     only_headline_data = np.genfromtxt(out_headline_path + str(x) + ".csv", dtype=float, delimiter=',')
#     # initialize combined data with extra column
#     if article_text_lines_count == 1:
#         out_combined_data = np.append(only_caption_data, np.zeros(article_text_lines_count))
#     else:
#         out_combined_data = np.c_[only_caption_data, np.zeros(np.size(text_data, 0))]
#
#     if article_text_lines_count == 1:
#         out_combined_data[-1] = only_headline_data[-1]
#     else:
#         for i in range(0, article_text_lines_count):
#             out_combined_data[i][-1] = only_headline_data[i][-1]
#     np.savetxt(out_combined_path + str(x) + ".csv", out_combined_data, delimiter=",")

text_path = "F:/Text Summarization/CNN Corpus/2_article text/csv/training/"
caption_path = "F:/Text Summarization/CNN Corpus/4_image captions/csv/training/"
summary_path = "F:/Text Summarization/CNN Corpus/5_highlights/csv/training/"
headline_path = "F:/Text Summarization/CNN Corpus/3_title/csv/training/"
out_caption_path = "F:/Text Summarization/Neural Network Dataset/3. Caption/training/"
out_caption_data = np.array([])
out_summary_path = "F:/Text Summarization/Neural Network Dataset/Output/training/"
out_summary_data = np.array([])
out_headline_path = "F:/Text Summarization/Neural Network Dataset/2. Headline/training/"
out_headline_data = np.array([])
out_combined_data = np.array([])
out_combined_path = "F:/Text Summarization/Neural Network Dataset/4. Caption and Headline/training/"

for x in range(1, 83565):
    print("training" + str(x))
    text_data = np.genfromtxt(text_path + str(x) + ".csv", dtype=float, delimiter=',')

    article_text_lines_count = 1
    if text_data.ndim == 2:
        article_text_lines_count = text_data.shape[0]
    # empty summary data
    out_summary_data = np.zeros(article_text_lines_count)
    # initialize caption data with extra column
    if article_text_lines_count == 1:
        out_caption_data = np.append(text_data, np.zeros(article_text_lines_count))
    else:
        out_caption_data = np.c_[text_data, np.zeros(np.size(text_data, 0))]
    # initialize headline data with extra column
    if article_text_lines_count == 1:
        out_headline_data = np.append(text_data, np.zeros(article_text_lines_count))
    else:
        out_headline_data = np.c_[text_data, np.zeros(np.size(text_data, 0))]

    caption_data = np.genfromtxt(caption_path + str(x) + ".csv", dtype=float, delimiter=',')
    if caption_data.size > 0:
        caption_lines_count = 1
        if caption_data.ndim == 2:
            caption_lines_count = caption_data.shape[0]
        for i in range(0, caption_lines_count):
            matched_index = -1;
            matched_value = math.inf
            for j in range(0, article_text_lines_count):
                dist = spatial.distance.cosine(caption_data[i], text_data[j])
                if dist < matched_value:
                    matched_index = j
                    matched_value = dist
            if article_text_lines_count == 1:
                out_caption_data[-1] = 0.569
            else:
                out_caption_data[matched_index, -1] = 0.569
    np.savetxt(out_caption_path + str(x) + ".csv", out_caption_data, delimiter=",")

    summary_data = np.genfromtxt(summary_path + str(x) + ".csv", dtype=float, delimiter=',')
    if summary_data.size > 0:
        summary_lines_count = 1
        if summary_data.ndim == 2:
            summary_lines_count = summary_data.shape[0]
        for i in range(0, summary_lines_count):
            matched_index = -1;
            matched_value = math.inf
            for j in range(0, article_text_lines_count):
                dist = spatial.distance.cosine(summary_data[i], text_data[j])
                if dist < matched_value:
                    matched_index = j
                    matched_value = dist
            out_summary_data[matched_index] = 1
    else:
        print('no summary?')
    np.savetxt(out_summary_path + str(x) + ".csv", out_summary_data, delimiter=",")

    headline_data = np.genfromtxt(headline_path + str(x) + ".csv", dtype=float, delimiter=',')
    if headline_data.size > 0:
        headline_lines_count = 1
        if headline_data.ndim == 2:
            headline_lines_count = headline_data.shape[0]
        for i in range(0, headline_lines_count):
            matched_index = -1;
            matched_value = math.inf
            for j in range(0, article_text_lines_count):
                dist = spatial.distance.cosine(headline_data[i], text_data[j])
                if dist < matched_value:
                    matched_index = j
                    matched_value = dist
            if article_text_lines_count == 1:
                out_headline_data[-1] = 0.569
            else:
                out_headline_data[matched_index, -1] = 0.569
    np.savetxt(out_headline_path + str(x) + ".csv", out_headline_data, delimiter=",")

    only_caption_data = np.genfromtxt(out_caption_path + str(x) + ".csv", dtype=float, delimiter=',')
    only_headline_data = np.genfromtxt(out_headline_path + str(x) + ".csv", dtype=float, delimiter=',')
    # initialize combined data with extra column
    if article_text_lines_count == 1:
        out_combined_data = np.append(only_caption_data, np.zeros(article_text_lines_count))
    else:
        out_combined_data = np.c_[only_caption_data, np.zeros(np.size(text_data, 0))]

    if article_text_lines_count == 1:
        out_combined_data[-1] = only_headline_data[-1]
    else:
        for i in range(0, article_text_lines_count):
            out_combined_data[i][-1] = only_headline_data[i][-1]
    np.savetxt(out_combined_path + str(x) + ".csv", out_combined_data, delimiter=",")
