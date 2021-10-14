import codecs
import csv
import os
import datetime
import time


def create_amazonbeauty_corpus(part):  # https://www.kaggle.com/skillsmuggler/amazon-ratings
    # Because the user id and item id aren't int, we make sure to pass every id to an int id.
    file_out = './data/amazonbeauty_corpus_small_' + part + '.csv'  # The file of the output, the ready corpus
    data_dir = './archive/amazonbeauty'  # The directory of the files.
    count = 0
    user_to_num = {}
    user_counter = 1
    users_amount = {}
    id_to_num = {}
    id_counter = 1
    items_amount = {}
    rating_counter = 0
    with open(file_out, 'w', newline='') as w_file:
        for file in sorted(os.listdir(data_dir)):
            print(file)
            with codecs.open(os.path.join(data_dir, file), 'rU', encoding="utf8") as r_file:
                for line in r_file:
                    if count < 1:
                        count += 1
                    elif count >= 1:
                        line = line.split(',')
                        if line[0] not in user_to_num.keys():
                            user_to_num[line[0]] = user_counter
                            user_counter += 1
                        user = user_to_num[line[0]]
                        if user not in users_amount.keys():
                            users_amount[user] = 1
                        else:
                            users_amount[user] += 1
                        if line[1] not in id_to_num.keys():
                            id_to_num[line[1]] = id_counter
                            id_counter += 1
                        cus_id = id_to_num[line[1]]
                        if cus_id not in items_amount.keys():
                            items_amount[cus_id] = 1
                        else:
                            items_amount[cus_id] += 1
                        rating = line[2]
                        rating_counter += 1
    keep_items = set()
    if part == 'light':
        for item in items_amount.keys():
            if items_amount[item] >= 10:
                keep_items.add(item)
    elif part == 'heavy':
        for item in items_amount.keys():
            if items_amount[item] <= 100:
                keep_items.add(item)
    count = 0
    with open(file_out, 'w', newline='') as w_file:
        writer = csv.writer(w_file, delimiter=',', quotechar='"', escapechar='\n', quoting=csv.QUOTE_NONE)
        for file in sorted(os.listdir(data_dir)):
            with codecs.open(os.path.join(data_dir, file), 'rU', encoding="utf8") as r_file:
                for line in r_file:
                    if count < 1:
                        count += 1
                    elif count >= 1:
                        line = line.split(',')
                        user = user_to_num[line[0]]
                        cus_id = id_to_num[line[1]]
                        rating = line[2]
                        date = line[3][:-1]
                        if cus_id in keep_items:
                            writer.writerow([user, cus_id, rating, date])

create_amazonbeauty_corpus('heavy')
create_amazonbeauty_corpus('light')
"""
All dataset:
============
# Users = 1,210,272
# Items = 249,275
# Ratings = 2,023,070
"""