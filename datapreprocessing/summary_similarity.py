import math

import numpy as np
from scipy import spatial

text_path = "/home/kailashkarthik/Text Summarization/CNN Corpus/2_article text/csv/training/"
summary_path = "/home/kailashkarthik/Text Summarization/CNN Corpus/5_highlights/csv/training/"
out_summary_path = "/home/kailashkarthik/Text Summarization/Neural Network Dataset/3. Captcha/training/output/"
out_summary_data = np.array([])

for x in range(1, 83564):
    print(x)
    text_data = np.genfromtxt(text_path + str(x) + ".csv", dtype=float, delimiter=',')
    summary_data = np.genfromtxt(summary_path + str(x) + ".csv", dtype=float, delimiter=',')
    out_summary_data = np.zeros(np.size(text_data, 0))
    if summary_data.size > 0:
        m = 1
        if summary_data.ndim == 2:
            m = summary_data.shape[0]
        for i in range(0, m):
            matched_index = -1;
            matched_value = math.inf
            p = text_data.shape[0]
            for j in range(0, p):
                dist = spatial.distance.cosine(summary_data[i], text_data[j])
                if dist < matched_value:
                    matched_index = j
                    matched_value = dist
            out_summary_data[matched_index] = 1
    np.savetxt(out_summary_path + str(x) + ".csv", out_summary_data, delimiter=",")
