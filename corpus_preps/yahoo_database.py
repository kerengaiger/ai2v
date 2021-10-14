import codecs
import csv
import os
import datetime
import time


def create_yahoo_all_corpus():  # https://webscope.sandbox.yahoo.com/catalog.php?datatype=c&did=48
    file_out = './data/yahoo_all_corpus.csv'  # The file of the output, the ready corpus
    data_dir = './archive/yahoo_all'  # The directory of the files.
    user_counter = 0
    item_counter = 0
    item_set = set()
    items_amount = {}
    line_counter = 0
    last_user = ''
    start_date = datetime.date(2000, 1, 1)
    with open(file_out, 'w', newline='') as w_file:
        writer = csv.writer(w_file, delimiter=',', quotechar='"', escapechar='\n', quoting=csv.QUOTE_NONE)
        for file in sorted(os.listdir(data_dir)):
            print(file)
            with codecs.open(os.path.join(data_dir, file), 'rU') as r_file:
                for line in r_file:
                    if user_counter >= 20000:
                        break
                    line = line.split('|')
                    if len(line) != 1:
                        user = line[0]
                        user_counter += 1
                    elif len(line) == 1:
                        line = line[0].split('\t')
                        cus_id = line[0]
                        if cus_id not in item_set:
                            item_set.add(cus_id)
                            items_amount[cus_id] = 1
                            item_counter += 1
                        else:
                            items_amount[cus_id] += 1
                        line_counter += 1
                        if (line_counter % 500000) == 0:
                            print(line_counter)
            user_counter = 0
            line_counter = 0
            top_items = sorted(items_amount.items(), key=lambda x: x[1], reverse=True)[:30000]
            top_items = [item[0] for item in top_items]
            top_items = set(top_items)
            with codecs.open(os.path.join(data_dir, file), 'rU') as r_file:
                print("staring with the file")
                for line in r_file:
                    if user_counter >= 20000:
                        break
                    line = line.split('|')
                    if len(line) != 1:
                        user = line[0]
                        user_counter += 1
                    elif len(line) == 1:
                        line = line[0].split('\t')
                        cus_id = line[0]
                        if cus_id not in top_items:
                            continue
                        rating = float(line[1])/20
                        if last_user != user:
                            last_user = user
                            date_time = start_date
                        else:
                            date_time = date_time + datetime.timedelta(days=1)
                        date = time.mktime(datetime.datetime.strptime(str(date_time), "%Y-%m-%d").timetuple())
                        writer.writerow([user, cus_id, rating, int(date)])
                        line_counter += 1
                        if (line_counter % 500000) == 0:
                            print(line_counter)
    print("user_counter")
    print(user_counter)
    print()
    print("item_counter")
    print(item_counter)
    print()
    print("line_counter")
    print(line_counter)


create_yahoo_all_corpus()
"""
All dataset:
============
#Users=5014136
#Items=1158226
#Ratings=1329499381
(#TrainRatings=1279358021, #ValidationRatings=20056544, #TestRatings=30084816)

First line for a user is formatted as: <UsedId>|<#UserRatings>
Each of the next <#UserRatings> lines describes a single rating by <UsedId>, sorted in chronological order.
Rating line format is: <ItemId>\t<Score>\t<Time>
The scores are integers lying between 0 and 100.  ---should be between 0 to 5
"""