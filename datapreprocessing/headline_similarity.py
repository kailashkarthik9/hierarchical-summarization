import math

import numpy as np
from scipy import spatial

text_path = "/home/kailashkarthik/Text Summarization/CNN Corpus/2_article text/csv/test/"
headline_path = "/home/kailashkarthik/Text Summarization/CNN Corpus/3_title/csv/test/"
out_headline_path = "/home/kailashkarthik/Text Summarization/Neural Network Dataset/3. Captcha/test/input/"
out_headline_data = np.array([])

for x in range(1, 1091):
    print(x)
    text_data = np.genfromtxt(text_path + str(x) + ".csv", dtype=float, delimiter=',')
    headline_data = np.genfromtxt(headline_path + str(x) + ".csv", dtype=float, delimiter=',')
    out_headline_data = np.c_[text_data, np.zeros(np.size(text_data, 0))]
    if headline_data.size > 0:
        m = 1
        if headline_data.ndim == 2:
            m = headline_data.shape[0]
        for i in range(0, m):
            matched_index = -1;
            matched_value = math.inf
            p = text_data.shape[0]
            for j in range(0, p):
                dist = spatial.distance.cosine(headline_data[i], text_data[j])
                if dist < matched_value:
                    matched_index = j
                    matched_value = dist
            out_headline_data[matched_index, 300] = 0.569
    np.savetxt(out_headline_path + str(x) + ".csv", out_headline_data, delimiter=",")

text_path = "/home/kailashkarthik/Text Summarization/CNN Corpus/2_article text/csv/training/"
headline_path = "/home/kailashkarthik/Text Summarization/CNN Corpus/3_title/csv/training/"
out_headline_path = "/home/kailashkarthik/Text Summarization/Neural Network Dataset/3. Captcha/training/input/"
out_headline_data = np.array([])

for x in range(1, 83564):
    print(x)
    text_data = np.genfromtxt(text_path + str(x) + ".csv", dtype=float, delimiter=',')
    headline_data = np.genfromtxt(headline_path + str(x) + ".csv", dtype=float, delimiter=',')
    out_headline_data = np.c_[text_data, np.zeros(np.size(text_data, 0))]
    if headline_data.size > 0:
        m = 1
        if headline_data.ndim == 2:
            m = headline_data.shape[0]
        for i in range(0, m):
            matched_index = -1;
            matched_value = math.inf
            p = text_data.shape[0]
            for j in range(0, p):
                dist = spatial.distance.cosine(headline_data[i], text_data[j])
                if dist < matched_value:
                    matched_index = j
                    matched_value = dist
            out_headline_data[matched_index, 300] = 0.569
    np.savetxt(out_headline_path + str(x) + ".csv", out_headline_data, delimiter=",")
