import glob
import os

import nltk

path = "/home/kailashkarthik/cnn/1_article text/training/*.txt"
write_path = "/home/kailashkarthik/cnn/2_article text/training/"
files = glob.glob(path)
for file in files:
    f = open(file, 'r')
    text = f.read()
    sentences = nltk.sent_tokenize(text)
    file_name = os.path.basename(f.name)
    file_out = open(write_path + file_name, 'w+')
    for sentence in sentences:
        file_out.write(sentence + "\n")
    f.close()
    file_out.close()
