import codecs
import csv
import os

file_out = 'netflix_corpus.csv'
data_dir = './archive'

for file in os.listdir(data_dir):
    with codecs.open(file_out, mode='w') as w_file:
        writer = csv.writer(w_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with codecs.open(os.path.join(data_dir, file), 'r', encoding='utf-8') as r_file:
            for line in r_file:
                if ',' not in line:
                    user = int(line.split(':')[0])
                else:
                    cus_id, rating, date = line.split(',')
                    writer.writerow([user, cus_id, rating, date.replace('\n', '')])
