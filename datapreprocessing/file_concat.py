path = "/home/kailashkarthik/cnn/2_article text/training/"
write_path = "/home/kailashkarthik/cnn/2_article text/training_concat.txt"
text = ""
for x in range(1, 83565):
    f = open(path + str(x) + ".txt", 'r')
    text = text + f.read() + "\n"
file_out = open(write_path, 'w+')
file_out.write(text + "\n")
f.close()
file_out.close()
