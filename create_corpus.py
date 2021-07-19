import codecs
import csv
import os
from random import randrange
import datetime
import time


# MovieLens: https://grouplens.org/datasets/movielens/


def create_netflix_corpus():  # https://www.kaggle.com/netflix-inc/netflix-prize-data
    file_out = './data/netflix_corpus.csv'  # The file of the output, the ready corpus
    data_dir = './archive/netflix'  # The directory of the files.
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


def create_moviesdat_corpus():  # https://www.kaggle.com/rounakbanik/the-movies-dataset
    file_out = './data/moviesdat_corpus.csv'  # The file of the output, the ready corpus
    data_dir = './archive/moviesdat'  # The directory of the files.
    count = 0
    with codecs.open(file_out, mode='w') as w_file:
        writer = csv.writer(w_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for file in os.listdir(data_dir):
            print(file)
            with codecs.open(os.path.join(data_dir, file), 'rU') as r_file:
                for line in r_file:
                    if count < 1:
                        count += 1
                    elif count >= 1:
                        line = line.split(',')
                        user = line[0]
                        cus_id = line[1]
                        rating = line[2]
                        date = line[3][:-1]
                        writer.writerow([user, cus_id, rating, date])


def create_yahoo_corpus():  # https://webscope.sandbox.yahoo.com/catalog.php?datatype=c&did=48
    # We create a random timestamp.
    file_out = './data/yahoo_corpus.csv'  # The file of the output, the ready corpus
    data_dir = './archive/yahoo'  # The directory of the files.
    start_date = datetime.date(2000, 1, 1)
    with codecs.open(file_out, mode='w') as w_file:
        writer = csv.writer(w_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for file in os.listdir(data_dir):
            print(file)
            with codecs.open(os.path.join(data_dir, file), 'rU') as r_file:
                for line in r_file:
                    line = line.split('|')
                    if len(line) != 1:
                        user = line[0]
                    elif len(line) == 1:
                        line = line[0].split('\t')
                        cus_id = line[0]
                        rating = line[1]
                        random_number_of_days = randrange(10000)
                        random_date = str(start_date + datetime.timedelta(days=random_number_of_days))
                        date = time.mktime(datetime.datetime.strptime(random_date, "%Y-%m-%d").timetuple())
                        writer.writerow([user, cus_id, rating, int(date)])


def create_goodbooks_corpus():  # https://www.kaggle.com/zygmunt/goodbooks-10k
    # The data is sorted by time so we create a timestamp accordingly.
    file_out = './data/goodbooks_corpus.csv'  # The file of the output, the ready corpus
    data_dir = './archive/goodbooks'  # The directory of the files.
    count = 0
    last_cus = '0'
    with codecs.open(file_out, mode='w') as w_file:
        writer = csv.writer(w_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for file in os.listdir(data_dir):
            print(file)
            with codecs.open(os.path.join(data_dir, file), 'rU') as r_file:
                for line in r_file:
                    if count < 1:
                        count += 1
                    elif count >= 1:
                        line = line.split(',')
                        user = line[1]
                        cus_id = line[0]
                        rating = line[2][:-1]
                        if last_cus != cus_id:
                            last_cus = cus_id
                            start_date = datetime.date(2000, 1, 1)
                            date_time = start_date
                        else:
                            date_time = date_time + datetime.timedelta(days=1)
                        date = time.mktime(datetime.datetime.strptime(str(date_time), "%Y-%m-%d").timetuple())
                        print("row")
                        print([user, cus_id, rating, int(date)])
                        writer.writerow([user, cus_id, rating, int(date)])


def create_booksrec_corpus():  # https://www.kaggle.com/arashnic/book-recommendation-dataset
    # We create a random timestamp.
    # Because the item id isn't int, we make sure to pass every id to an int id.
    file_out = './data/booksrec_corpus.csv'  # The file of the output, the ready corpus
    data_dir = './archive/booksrec'  # The directory of the files.
    count = 0
    start_date = datetime.date(2000, 1, 1)
    id_to_num = {}
    id_counter = 1
    with codecs.open(file_out, mode='w') as w_file:
        writer = csv.writer(w_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for file in os.listdir(data_dir):
            print(file)
            with codecs.open(os.path.join(data_dir, file), 'rU', encoding="utf8") as r_file:
                for line in r_file:
                    if count < 1:
                        count += 1
                    elif count >= 1:
                        line = line.split(',')
                        user = line[0]
                        if line[1] not in id_to_num.keys():
                            id_to_num[line[1]] = id_counter
                            id_counter += 1
                        cus_id = id_to_num[line[1]]
                        rating = line[2][:-1]
                        random_number_of_days = randrange(10000)
                        random_date = str(start_date + datetime.timedelta(days=random_number_of_days))
                        date = time.mktime(datetime.datetime.strptime(random_date, "%Y-%m-%d").timetuple())
                        writer.writerow([user, cus_id, rating, int(date)])


def create_animerec_corpus():  # https://www.kaggle.com/CooperUnion/anime-recommendations-database
    # We create a random timestamp.
    file_out = './data/animerec_corpus.csv'  # The file of the output, the ready corpus
    data_dir = './archive/animerec'  # The directory of the files.
    count = 0
    start_date = datetime.date(2000, 1, 1)
    with codecs.open(file_out, mode='w') as w_file:
        writer = csv.writer(w_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for file in os.listdir(data_dir):
            print(file)
            with codecs.open(os.path.join(data_dir, file), 'rU') as r_file:
                for line in r_file:
                    if count < 1:
                        count += 1
                    elif count >= 1:
                        line = line.split(',')
                        user = line[0]
                        cus_id = line[1]
                        rating = line[2][:-1]
                        random_number_of_days = randrange(10000)
                        random_date = str(start_date + datetime.timedelta(days=random_number_of_days))
                        date = time.mktime(datetime.datetime.strptime(random_date, "%Y-%m-%d").timetuple())
                        writer.writerow([user, cus_id, rating, int(date)])


def create_animerec20_corpus():  # https://www.kaggle.com/hernan4444/anime-recommendation-database-2020
    # We create a random timestamp.
    file_out = './data/animerec20_corpus.csv'  # The file of the output, the ready corpus
    data_dir = './archive/animerec20'  # The directory of the files.
    count = 0
    start_date = datetime.date(2000, 1, 1)
    with codecs.open(file_out, mode='w') as w_file:
        writer = csv.writer(w_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for file in os.listdir(data_dir):
            print(file)
            with codecs.open(os.path.join(data_dir, file), 'rU') as r_file:
                for line in r_file:
                    if count < 1:
                        count += 1
                    elif count >= 1:
                        line = line.split(',')
                        user = line[0]
                        cus_id = line[1]
                        rating = line[2][:-1]
                        random_number_of_days = randrange(10000)
                        random_date = str(start_date + datetime.timedelta(days=random_number_of_days))
                        date = time.mktime(datetime.datetime.strptime(random_date, "%Y-%m-%d").timetuple())
                        writer.writerow([user, cus_id, rating, int(date)])


def main():
    #create_netflix_corpus()
    #create_moviesdat_corpus()
    #create_yahoo_corpus()
    create_goodbooks_corpus()
    #create_booksrec_corpus()
    #create_animerec_corpus()
    #create_animerec20_corpus()
    # https://www.kaggle.com/skillsmuggler/amazon-ratings

if __name__ == '__main__':
    main()
