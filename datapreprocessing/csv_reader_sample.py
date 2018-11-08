import csv

csv_file = open("/home/kailashkarthik/cnn/2_article text/csv/test/1.csv", mode="r")
csv_reader = csv.DictReader(csv_file)
for row in csv_reader:
    print(row)
