import codecs
import csv
import os

file_out = './data/netflix_corpus.csv'
data_dir = './archive'

with codecs.open(file_out, mode='w') as w_file:
    writer = csv.writer(w_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for file in os.listdir(data_dir):
        print(file)
        with codecs.open(os.path.join(data_dir, file), 'rU') as r_file:
            for line in r_file:
                if ',' not in line:
                    user = int(line.split(':')[0])

                else:
                    cus_id, rating, date = line.split(',')
                    writer.writerow([user, cus_id, rating, date])
