import os
import numpy as np
import torch
import pandas as pd
from abc import ABC, abstractmethod
import pickle
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
from transformers import BertTokenizer


TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
UNLABELED_SET = "unlabeled"
DEV32_SET = "dev32"

def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_dataset(args, data, labelled: bool = True):
    # features = convert_examples_to_features_for_earningcall(data, labeled=labelled)
    feature_dict = {}
    for k, v in data.items():
        if k in ['mlm_input_ids', 'sent_is_ratio', 'labels_is_ratio', 'labels_mlm', 'head_nd_input_ids', 'rel_nd_input_ids', 'tail_nd_input_ids']:
            feature_dict[k] = nn.utils.rnn.pad_sequence(v, batch_first=True, padding_value=0)
        elif k in ['position']:
            feature_dict[k] = nn.utils.rnn.pad_sequence(v, batch_first=True, padding_value=-1)
            
    return DictDataset(**feature_dict)

def convert_examples_to_features_for_earningcall(examples, labeled):
    label_list = ["0", "1"]
    label_map = {label: i for i, label in enumerate(label_list)}
        
    examples['label_avg3days'] = []
    examples['label_avg7days'] = []
    examples['label_avg15days'] = []
    examples['label_avg30days'] = []
    
    examples['label_single3days'] = []
    examples['label_single7days'] = []
    examples['label_single15days'] = []
    examples['label_single30days'] = []
    
    # examples['mlm_labels'] = []
    # examples['input_ids'] = []
    # examples['attention_mask'] = []
    # examples['token_type_ids'] = []
    # examples['block_flag'] = []
    
    for idx in range(len(examples['company_name'])):

        examples['label_avg3days'] += [label_map[str(int(examples['y_avg3days'][idx] >= 0))]]
        examples['label_avg7days'] += [label_map[str(int(examples['y_avg7days'][idx] >= 0))]]
        examples['label_avg15days'] += [label_map[str(int(examples['y_avg15days'][idx] >= 0))]]
        examples['label_avg30days'] += [label_map[str(int(examples['y_avg30days'][idx] >= 0))]]
        
        examples['label_single3days'] += [label_map[str(int(examples['y_single3days'][idx] >= 0))]]
        examples['label_single7days'] += [label_map[str(int(examples['y_single7days'][idx] >= 0))]]
        examples['label_single15days'] += [label_map[str(int(examples['y_single15days'][idx] >= 0))]]
        examples['label_single30days'] += [label_map[str(int(examples['y_single30days'][idx] >= 0))]]
        
    return examples


def load_examples(args, task, data_dir: str, set_type: str, *_, num_examples: int = None,
                  seed: int = 42, split_examples_evenly: bool = False):
    """Load examples for a given task."""

    def eq_div(N, i):
        """ Equally divide N examples among i buckets. For example, `eq_div(12,3) = [4,4,4]`. """
        return [] if i <= 0 else [N // i + 1] * (N % i) + [N // i] * (i - N % i)

    num_examples_per_label = None
    if split_examples_evenly:
        num_examples_per_label = eq_div(
            num_examples, len(PROCESSORS[task]().get_labels()))
        num_examples = None

    SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET, DEV32_SET]

    assert (num_examples is not None) ^ (num_examples_per_label is not None), \
        "Exactly one of 'num_examples' and 'num_examples_per_label' must be set."
    assert (not set_type == UNLABELED_SET) or (num_examples is not None), \
        "For unlabeled data, 'num_examples_per_label' is not allowed"
        
    if task == 'EarningCall':
        processor = PROCESSORS[task](args, data_dir)
    else:
        processor = PROCESSORS[task]()
        
    ex_str = f"num_examples={num_examples}" if num_examples is not None \
        else f"num_examples_per_label={num_examples_per_label}"
    args.logger.debug(
        f"Creating features from dataset file at {data_dir} ({ex_str}, set_type={set_type})"
    )

    if set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir)
    else:
        raise ValueError(
            f"'set_type' must be one of {SET_TYPES}, got '{set_type}' instead")
    
    return examples


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()

        assert all(next(iter(tensors.values())).size(0) == tensor.size(0)
                   for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)


class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading train/dev32/dev/test/unlabeled examples for a given task.
    """

    @abstractmethod
    def get_train_examples(self, data_dir):
        """Get a collection of `InputExample`s for the train set."""
        pass


class EarningCallProcessor(DataProcessor):
    """Processor for the Stock data set (GLUE)."""

    def __init__(self, args, data_dir):
        
        self.args = args
        
        processed_ptm_path = data_dir + 'processed_dataset_for_kept' 
        processed_ptm_list = os.listdir(processed_ptm_path)
        
        sizes_list = []
        for ptm_dataset_name in tqdm(processed_ptm_list):
            sizes_list += [(ptm_dataset_name, os.path.getsize(processed_ptm_path + '/' + ptm_dataset_name))]
        ptm_dataset_list = sorted(sizes_list, key=lambda x: x[1])
        
        self.ptm_dataset = []
        for ptm_dataset_name, _ in tqdm(ptm_dataset_list):
            if 'ent_rel' in ptm_dataset_name:
                continue
            print(ptm_dataset_name)
            self.ptm_dataset += pickle.load(open(processed_ptm_path + '/' + ptm_dataset_name, 'rb'))
            
        self.new_tokens = []
        with open(data_dir + '/supports/financial_metrics.txt', 'r') as f:
            for line in f.readlines():
                self.new_tokens += [line.strip()]
        
        with open(data_dir + '/supports/financial_metrics_supplyment.txt', 'r') as f:
            for line in f.readlines():
                self.new_tokens += [line.strip()]
        
        self.args.tokenizer.add_tokens(self.new_tokens)
            
    def generate_padded_token_embedding(self, max_sent=520):
        
        token_embedding = self.text_token_dict
        output_dct = {}
        for k in tqdm(token_embedding.keys()):
            out_for_on_text = torch.tensor(token_embedding[k])
            num_sent, sent_max_len, ptm_size = out_for_on_text.shape
            if max_sent > num_sent:
                padded_out_for_on_text = torch.cat([out_for_on_text, torch.zeros(max_sent - num_sent, sent_max_len, ptm_size)], dim=0)
            else:
                padded_out_for_on_text = out_for_on_text[:max_sent, :, :]
            output_dct[k] = padded_out_for_on_text
            
        return output_dct
    
    def get_train_examples(self):
        return self._create_examples()

    def PTMData(self):
        
        out_dict = {}
        out_dict['mlm_input_ids'] = []
        out_dict['sent_is_ratio']=[]
        out_dict['labels_mlm']=[]
        out_dict['labels_is_ratio']=[]
        out_dict['head_nd_input_ids']=[]
        out_dict['rel_nd_input_ids']=[]
        out_dict['tail_nd_input_ids']=[]
        out_dict['position'] = []
        need_sents = []
        
        miss_i = 0
        
        for item in tqdm(self.ptm_dataset):
            
            if item['mlm_labels'] == []:
                continue
            
            position = []
            head_nd_input_ids = []
            rel_nd_input_ids = []
            tail_nd_input_ids = []
            for ent in item['entities']:
                if ent['triples'] == [] or ent['desc'] == []:
                    continue
                one_ent_head_nd_input_ids = []
                one_ent_rel_nd_input_ids = []
                one_ent_tail_nd_input_ids = []
                position += [torch.tensor(ent['position'])]
                for trp, desc in zip(ent['triples'], ent['desc']):
                    head_name_desc = self.args.tokenizer(ent['name'] + ' ' + self.args.tokenizer.sep_token + ' ' + ent['description'],
                                                                padding='max_length', max_length=self.args.entity_max_length, truncation=True, return_tensors='pt')
                    one_ent_head_nd_input_ids += [head_name_desc['input_ids']]

                    rel_name_desc = self.args.tokenizer(trp[1] + ' ' + self.args.tokenizer.sep_token + ' ' + desc[1],
                                                                padding='max_length', max_length=self.args.entity_max_length, truncation=True, return_tensors='pt')
                    one_ent_rel_nd_input_ids += [rel_name_desc['input_ids']]
                    
                    tail_name_desc = self.args.tokenizer(trp[2] + ' ' + self.args.tokenizer.sep_token + ' ' + desc[2],
                                                                padding='max_length', max_length=self.args.entity_max_length, truncation=True, return_tensors='pt')
                    one_ent_tail_nd_input_ids += [tail_name_desc['input_ids']]
                
                head_nd_input_ids += [F.pad(nn.utils.rnn.pad_sequence(one_ent_head_nd_input_ids, batch_first=True, padding_value=0).squeeze(1), (0, 0, 0, 6-len(ent['triples'])), 'constant', 0)]
                rel_nd_input_ids += [F.pad(nn.utils.rnn.pad_sequence(one_ent_rel_nd_input_ids, batch_first=True, padding_value=0).squeeze(1), (0, 0, 0, 6-len(ent['triples'])), 'constant', 0)]
                tail_nd_input_ids += [F.pad(nn.utils.rnn.pad_sequence(one_ent_tail_nd_input_ids, batch_first=True, padding_value=0).squeeze(1), (0, 0, 0, 6-len(ent['triples'])), 'constant', 0)]
            
            if head_nd_input_ids == [] or rel_nd_input_ids == [] or tail_nd_input_ids == []:
                continue
            
            position = nn.utils.rnn.pad_sequence(position, batch_first=True, padding_value=-1)
            position_in = F.pad(position, (0, 2-position.shape[-1]), 'constant', -1)
            posi_mask = (position_in != -1).float()
            bidx_posi, _ = posi_mask.nonzero(as_tuple=True)
            
            labels_is_ratio = torch.tensor([0.0] + item['is_ratio'] + [0.0])
            bidx = (labels_is_ratio == 1).nonzero(as_tuple=True)[0]
            
            if len(bidx_posi) != len(bidx):
                print('miss match position and is_ratios, count is {}'.format(miss_i))
                miss_i += 1
                continue
        
            mlm_input = self.args.tokenizer(item['mlm_sent'], padding='max_length', max_length=self.args.ratio_max_length, truncation=True, is_split_into_words=True, return_tensors='pt')
            lb_mlm_index = self.args.tokenizer.convert_tokens_to_ids(item['mlm_labels'])
            
            b_mask_posi, _ = (mlm_input['input_ids'] == 5).nonzero(as_tuple=True)   # 5 is mask id
            lb_bidx = (torch.tensor(lb_mlm_index) != 0).nonzero(as_tuple=True)[0]
            
            if len(b_mask_posi) != len(lb_bidx):
                print('miss match mask and label, count is {}'.format(miss_i))
                miss_i += 1
                continue
            
            out_dict['position'] += [position_in]
            out_dict['head_nd_input_ids'] += [nn.utils.rnn.pad_sequence(head_nd_input_ids, batch_first=True, padding_value=0)]
            out_dict['rel_nd_input_ids'] += [nn.utils.rnn.pad_sequence(rel_nd_input_ids, batch_first=True, padding_value=0)]
            out_dict['tail_nd_input_ids'] += [nn.utils.rnn.pad_sequence(tail_nd_input_ids, batch_first=True, padding_value=0)]

            
            out_dict['mlm_input_ids'] += [mlm_input['input_ids']]
            out_dict['labels_mlm'] += [torch.tensor(lb_mlm_index)]
            
            input_is_ratio = self.args.tokenizer(item['raw_sent'], padding='max_length', max_length=self.args.ratio_max_length, truncation=True, return_tensors='pt')
            out_dict['sent_is_ratio'] += [input_is_ratio['input_ids']]
            out_dict['labels_is_ratio'] += [labels_is_ratio]
            need_sents += [item['raw_sent']]
            
        return out_dict
    
    def _create_examples(self):
        
        data_dict = self.PTMData()
        self.args.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        return data_dict


PROCESSORS = {
    "EarningCall": EarningCallProcessor,
}