import math

import numpy as np
from scipy import spatial

text_path = "/home/kailashkarthik/Text Summarization/CNN Corpus/2_article text/csv/training/"
caption_path = "/home/kailashkarthik/Text Summarization/CNN Corpus/4_image captions/csv/training/"
out_caption_path = "/home/kailashkarthik/Text Summarization/Neural Network Dataset/3. Caption/training/"
out_caption_data = np.array([])

for x in range(1, 83564):
    print(x)
    text_data = np.genfromtxt(text_path + str(x) + ".csv", dtype=float, delimiter=',')
    caption_data = np.genfromtxt(caption_path + str(x) + ".csv", dtype=float, delimiter=',')
    out_caption_data = np.c_[text_data, np.zeros(np.size(text_data, 0))]
    if caption_data.size > 0:
        m = 1
        if caption_data.ndim == 2:
            m = caption_data.shape[0]
        for i in range(0, m):
            matched_index = -1;
            matched_value = math.inf
            p = text_data.shape[0]
            for j in range(0, p):
                dist = spatial.distance.cosine(caption_data[i], text_data[j])
                if dist < matched_value:
                    matched_index = j
                    matched_value = dist
            out_caption_data[matched_index, 300] = 0.569
    np.savetxt(out_caption_path + str(x) + ".csv", out_caption_data, delimiter=",")

text_path = "/home/kailashkarthik/Text Summarization/CNN Corpus/2_article text/csv/training/"
caption_path = "/home/kailashkarthik/Text Summarization/CNN Corpus/4_image captions/csv/training/"
out_caption_path = "/home/kailashkarthik/Text Summarization/Neural Network Dataset/3. Caption/training/input/"
out_caption_data = np.array([])

for x in range(1, 83564):
    print(x)
    text_data = np.genfromtxt(text_path + str(x) + ".csv", dtype=float, delimiter=',')
    caption_data = np.genfromtxt(caption_path + str(x) + ".csv", dtype=float, delimiter=',')
    out_caption_data = np.c_[text_data, np.zeros(np.size(text_data, 0))]
    if caption_data.size > 0:
        m = 1
        if caption_data.ndim == 2:
            m = caption_data.shape[0]
        for i in range(0, m):
            matched_index = -1;
            matched_value = math.inf
            p = text_data.shape[0]
            for j in range(0, p):
                dist = spatial.distance.cosine(caption_data[i], text_data[j])
                if dist < matched_value:
                    matched_index = j
                    matched_value = dist
            out_caption_data[matched_index, 300] = 0.569
    np.savetxt(out_caption_path + str(x) + ".csv", out_caption_data, delimiter=",")
