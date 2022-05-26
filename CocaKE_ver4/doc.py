import os
import json
import torch
import torch.utils.data.dataset
import random

from typing import Optional, List

from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from logger_config import logger

entity_dict = get_entity_dict()
if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)


class Example:

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        hr_encoded_inputs = _custom_tokenize(text=head_text,
                                             text_pair=self.relation)

        head_encoded_inputs = _custom_tokenize(text=head_text)

        tail_word = _parse_entity_name(self.tail)
        tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))

        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                'head_token_ids': head_encoded_inputs['input_ids'],
                'head_token_type_ids': head_encoded_inputs['token_type_ids'],
                'obj': self}


def vectorize_entity(ent_id, ent_type, relation=None):
    ent_desc = entity_dict.get_entity_by_id(ent_id).entity_desc
    if args.use_link_graph:
        if len(ent_desc.split()) < 20:
            ent_desc += ' ' + get_neighbor_desc(head_id=ent_id)
    ent_word = _parse_entity_name(entity_dict.get_entity_by_id(ent_id).entity)
    ent_text = _concat_name_desc(ent_word, ent_desc)
    if relation:
        ent_encoded_inputs = _custom_tokenize(text=ent_text,
                                             text_pair=relation)
    else:
        ent_encoded_inputs = _custom_tokenize(text=ent_text)
    
    return {
        ent_type+"_token_ids": ent_encoded_inputs["input_ids"],
        ent_type+"_token_type_ids": ent_encoded_inputs["token_type_ids"]
    }
        
    
        
            
    
    
    

class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, commonsense_path, head_ns_cnt, tail_ns_cnt, examples=None):
        self.path_list = path.split(',')
        self.task = task
        self.commonsense_path = commonsense_path
        self.ent_dom = json.load(open(os.path.join(commonsense_path,"ent_dom.json"), 'r', encoding='utf-8'))
        self.dom_ent = json.load(open(os.path.join(commonsense_path,"dom_ent.json"), 'r', encoding='utf-8'))
        self.rel2dom_h = json.load(open(os.path.join(commonsense_path,"rel2dom_h.json"), 'r', encoding='utf-8'))
        self.rel2dom_t = json.load(open(os.path.join(commonsense_path,"rel2dom_t.json"), 'r', encoding='utf-8'))
        self.rel2nn = json.load(open(os.path.join(commonsense_path,"rel2nn.json"), 'r', encoding='utf-8'))
        self.head_ns_cnt = head_ns_cnt
        self.tail_ns_cnt = tail_ns_cnt
        self.reverse_dict = {
          0 : 0,
          1 : 2,
          2 : 1,
          3 : 3
        }
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))
    # Relation_complexity given in CAKE
    # 0 : 1-1
    # 1 : 1-N
    # 2 : N-1
    # 3 : N-N
    def corrupt_head(self, example, head_ns_cnt):
        if example.relation[:8] == "inverse ":
            relation = example.relation[8:]
            rel_hc = self.rel2dom_t[str(relation)]
            rel2nn = self.reverse_dict[self.rel2nn[str(relation)]]
        else:
            relation = example.relation
            rel_hc = self.rel2dom_h[str(relation)]
            rel2nn = self.rel2nn[str(relation)]
        set_hc = set(rel_hc)
        h = []
        if rel2nn == 0 or rel2nn == 1:
            if str(example.head_id) not in self.ent_dom:
                for hc in rel_hc:
                    for ent in self.dom_ent[str(hc)]:
                        h.append(ent)
            else:
                for conc in self.ent_dom[str(example.head_id)]:
                    for ent in self.dom_ent[str(conc)]:
                        h.append(ent)
        else:
            if str(example.head_id) in self.ent_dom:
                set_ent_conc = set(self.ent_dom[str(example.head_id)])
            else:
                set_ent_conc = set([])
            set_diff = set_hc - set_ent_conc
            set_diff = list(set_diff)
            for conc in set_diff:
                for ent in self.dom_ent[str(conc)]:
                    h.append(ent)
        h = set(h)
        if len(h) > head_ns_cnt:
            return random.choices(list(h), k=head_ns_cnt)
        else:
            return list(h)
        
            
    def corrupt_tail(self, example, tail_ns_cnt):
        if example.relation[:8] == "inverse ":
            relation = example.relation[8:]
            rel_tc = self.rel2dom_h[str(relation)]
            rel2nn = self.reverse_dict[self.rel2nn[str(relation)]]
        else:
            relation = example.relation
            rel_tc = self.rel2dom_t[str(relation)]
            rel2nn = self.rel2nn[str(relation)]
        set_tc = set(rel_tc)
        t = []
        if rel2nn == 0 or rel2nn == 2:
            if str(example.tail_id) not in self.ent_dom:
                for tc in rel_tc:
                    for ent in self.dom_ent[str(tc)]:
                        t.append(ent)
            else:
                for conc in self.ent_dom[str(example.tail_id)]:
                    for ent in self.dom_ent[str(conc)]:
                        t.append(ent)
        else:
            if str(example.tail_id) in self.ent_dom:
                set_ent_conc = set(self.ent_dom[str(example.tail_id)])
            else:
                set_ent_conc = set([])
            set_diff = set_tc - set_ent_conc
            set_diff = list(set_diff)
            for conc in set_diff:
                for ent in self.dom_ent[str(conc)]:
                    t.append(ent)
        t = set(t)
        if len(t) > tail_ns_cnt:
            return random.choices(list(t), k=tail_ns_cnt)
        else:
            return list(t)
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        selected_example = self.examples[index]
        corrupted_head = self.corrupt_head(selected_example, self.head_ns_cnt)
        corrupted_tail = self.corrupt_tail(selected_example, self.tail_ns_cnt)
        neg_heads_vectorized = [vectorize_entity(head, "head") for head in corrupted_head]
        neg_tails_vectorized = [vectorize_entity(tail, "tail") for tail in corrupted_tail]
        neg_hrs_vectorized = [vectorize_entity(head, "hr", selected_example.relation) for head in corrupted_head]
        return {
            "simkgc": selected_example.vectorize(),
            "cake":{
                "heads": neg_heads_vectorized,
                "tails": neg_tails_vectorized,
                "hrs": neg_hrs_vectorized,    
            }
        }
        
    
    
def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    
    ################ simkgc ################
    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex["simkgc"]['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex["simkgc"]['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex["simkgc"]['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex["simkgc"]['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex["simkgc"]['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex["simkgc"]['head_token_type_ids']) for ex in batch_data],
        need_mask=False)
    ############### end of simkgc ############
    
    batch_exs = [ex["simkgc"]['obj'] for ex in batch_data]
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
