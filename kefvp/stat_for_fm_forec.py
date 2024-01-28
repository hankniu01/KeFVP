import spacy
import json
from tqdm import tqdm
import pandas as pd

nlp = spacy.load('en_core_web_sm')
    
radio_corpus = json.load(open('/your/project/path/data_process/needed_files/final_wiki_ids.json', 'r'))

radio_set = []
radio_set += [{'label': 'ebitda', 'pattern': 'ebitda'}]
for line in radio_corpus.keys():
    radio_set += [{'label': line.strip().lower(), 'pattern': line.strip().lower()}]

all_radio_set = radio_set
ruler = nlp.add_pipe('entity_ruler')
ruler.add_patterns(all_radio_set)

maec_path = '/your/project/path/raw_data/ReleasedDataset_mp3/'
base_dir = '/your/dataset/path/'
# dataset = '16'

for dataset in ['ec']:
    traindf= pd.read_csv(base_dir + "price_data/train_split_Avg_Series_WITH_LOG.csv")
    testdf=pd.read_csv(base_dir + "price_data/test_split_Avg_Series_WITH_LOG.csv")
    valdf=pd.read_csv(base_dir + "price_data/val_split_Avg_Series_WITH_LOG.csv")
    
    all_ent = {}
    for k, df in {'train': traindf, 'test': testdf, 'val': valdf}.items():
        all_ent[k] = {}
        for index, row in tqdm(df.iterrows()):
            all_path = maec_path + row['text_file_name'] + '/Text.txt'
            for line in open(all_path, 'r').readlines():
                doc = nlp(line.strip().lower())
                if doc.ents != ():
                    for ent in doc.ents:
                        if ent.label_.lower() not in all_ent[k].keys():
                            all_ent[k][ent.label_.lower()] = 1
                        else:
                            all_ent[k][ent.label_.lower()] += 1
    
    json.dump(all_ent, open('/your/project/path/save_fig/stat_fm_{}.json'.format(dataset), "w"))