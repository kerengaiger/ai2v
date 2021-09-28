import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

def bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus, amazon=False, precent=False):
    item_cnts = pickle.load(open(ic_file, 'rb'))
    itm_cnts_df = pd.DataFrame([item_cnts.keys(), item_cnts.values()]).T
    itm_cnts_df.columns = ['item_id', 'cnt']
    # itm_cnts_df.head(100)
    # print(len(itm_cnts_df))

    cnts_sorted = itm_cnts_df.sort_values(by=['cnt'], ascending=False)['cnt'].tolist()[:-2]
    # print(cnts_sorted)

    if amazon:
        items_sorted = [item for item in itm_cnts_df.sort_values(by=['cnt'], ascending=False)['item_id'].tolist()[:-2]]
        items_dict = {}
        counter = 0
        for item in items_sorted:
            if item not in items_dict.keys():
                items_dict[item] = counter
                counter += 1
        items_sorted = [items_dict[item] for item in items_sorted]
    else:
        items_sorted = [int(item) for item in
                        itm_cnts_df.sort_values(by=['cnt'], ascending=False)['item_id'].tolist()[:-2]]

    items_cnt_dict = {}
    count = 0
    samples_amount = 0
    for item in range(len(items_sorted)):
        items_cnt_dict[items_sorted[item]] = cnts_sorted[item]
        samples_amount += cnts_sorted[item]
        count += 1
    # print(count)
    # print(samples_amount)

    num_bins = 100
    bin_size = int(len(items_sorted) / num_bins)

    binned_items = [items_sorted[i * bin_size:(i + 1) * bin_size] for i in range(0, num_bins - 1)]
    last_bin = items_sorted[(num_bins - 1) * bin_size:]
    binned_items.append(last_bin)

    # print(len(binned_items))
    count = 0
    for part in binned_items:
        count += len(part)
        # print(len(part))
    # print(count)
    # print(len(items_sorted))

    cnt_per_bin = []
    for part in binned_items:
        bin_cnt = []
        for item in part:
            bin_cnt.append(items_cnt_dict[item])
        cnt_per_bin.append(bin_cnt)

    if precent == True:
        max_amount = []
        for part in binned_items:
            calc = (items_cnt_dict[part[0]] / samples_amount) * 100
            max_amount.append(calc)

        return max_amount
    else:
        max_amount = []
        for part in binned_items:
            max_amount.append(items_cnt_dict[part[0]])

        return max_amount


datasets_bins = []
datasets_bins_precent = []
labels = []

DATASET = 'movielens'
DATA_FOLD = f'corpus/{DATASET}_llo'
AI2V_OUT_FOLD = f'output/{DATASET}_ai2v_llo'
ic_file = f'{DATA_FOLD}/full_ic.dat'
full_corpus = f'{DATA_FOLD}/full_corpus.txt'

movielens_100bins = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus)
movielens_100bins_precent = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus, precent=True)
datasets_bins.append(movielens_100bins)
datasets_bins_precent.append(movielens_100bins_precent)
labels.append(DATASET)

DATASET = 'moviesdat'
DATA_FOLD = f'corpus/{DATASET}_llo'
AI2V_OUT_FOLD = f'output/{DATASET}_ai2v_llo'
ic_file = f'{DATA_FOLD}/full_ic.dat'
full_corpus = f'{DATA_FOLD}/full_corpus.txt'

moviesdat_100bins = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus)
moviesdat_100bins_precent = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus, precent=True)
datasets_bins.append(moviesdat_100bins)
datasets_bins_precent.append(moviesdat_100bins_precent)
labels.append(DATASET)

DATASET = 'netflix'
DATA_FOLD = f'corpus/{DATASET}_llo'
AI2V_OUT_FOLD = f'output/{DATASET}_ai2v_llo'
ic_file = f'{DATA_FOLD}/full_ic.dat'
full_corpus = f'{DATA_FOLD}/full_corpus.txt'

netflix_100bins = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus)
netflix_100bins_precent = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus, precent=True)
datasets_bins.append(netflix_100bins)
datasets_bins_precent.append(netflix_100bins_precent)
labels.append(DATASET)

DATASET = 'yahoo'
DATA_FOLD = f'corpus/{DATASET}_llo'
AI2V_OUT_FOLD = f'output/{DATASET}_ai2v_llo'
ic_file = f'{DATA_FOLD}/full_ic.dat'
full_corpus = f'{DATA_FOLD}/full_corpus.txt'

yahoo_100bins = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus)
yahoo_100bins_precent = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus, precent=True)
datasets_bins.append(yahoo_100bins)
datasets_bins_precent.append(yahoo_100bins_precent)
labels.append(DATASET)

DATASET = 'amazonbeauty'
DATA_FOLD = f'corpus/{DATASET}_llo'
AI2V_OUT_FOLD = f'output/{DATASET}_ai2v_llo'
ic_file = f'{DATA_FOLD}/full_ic.dat'
full_corpus = f'{DATA_FOLD}/full_corpus.txt'

amazonbeauty_100bins = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus, amazon=True)
amazonbeauty_100bins_precent = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus, amazon=True, precent=True)
datasets_bins.append(amazonbeauty_100bins)
datasets_bins_precent.append(amazonbeauty_100bins_precent)
labels.append(DATASET)

DATASET = 'goodbooks'
DATA_FOLD = f'corpus/{DATASET}_llo'
AI2V_OUT_FOLD = f'output/{DATASET}_ai2v_llo'
ic_file = f'{DATA_FOLD}/full_ic.dat'
full_corpus = f'{DATA_FOLD}/full_corpus.txt'

goodbooks_100bins = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus)
goodbooks_100bins_precent = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus, precent=True)
datasets_bins.append(goodbooks_100bins)
datasets_bins_precent.append(goodbooks_100bins_precent)
labels.append(DATASET)

DATASET = 'amazonbooks'
DATA_FOLD = f'corpus/{DATASET}_llo'
AI2V_OUT_FOLD = f'output/{DATASET}_ai2v_llo'
ic_file = f'{DATA_FOLD}/full_ic.dat'
full_corpus = f'{DATA_FOLD}/full_corpus.txt'

amazonbooks_100bins = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus, amazon=True)
amazonbooks_100bins_precent = bin_maker_100(DATASET, DATA_FOLD, AI2V_OUT_FOLD, ic_file, full_corpus, amazon=True, precent=True)
datasets_bins.append(amazonbooks_100bins)
datasets_bins_precent.append(amazonbooks_100bins_precent)
labels.append(DATASET)

number_of_bins = 100
linestyles = ["dashdot", "dashed", "solid", "dotted", "solid", "dashed", "dotted"]
markers = [",", ",", ",", "*", "^", "o", ","]
# 'b' as blue, 'g' as green, 'r' as red, 'c' as cyan, 'm' as magenta, 'y' as yellow, 'k' as black, 'w' as white
mpl.rcParams['axes.prop_cycle'] = cycler(color=['b', 'g', 'y', 'c', 'r', 'm', 'k'])
for data in range(len(datasets_bins_precent)):
    plt.plot([i for i in list(range(number_of_bins))], datasets_bins_precent[data], label=labels[data], marker=markers[data], linestyle=linestyles[data])
plt.xlabel('Popularity Bins (1% of the items)')
plt.ylabel('Item Prevalence in Dataset %')
plt.yscale('log')
plt.legend(bbox_to_anchor=(0.99, 0.99), borderaxespad=0.)  # loc='upper left'
plt.savefig('figures/Skew_levels.jpg', bbox_inches='tight')
plt.show()
