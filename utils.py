import os
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
from xml.dom import minidom
import random
import tarfile
import json

import openpyxl

from sklearn.model_selection import train_test_split



def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def parse_line(line):
    if len(line.split("\t<=>\t")) < 2:
        return

    utterance_data, intent_label = line.split("\t<=>\t")
    intent_label = intent_label.replace('\t', '.')
    items = utterance_data.split('\t')
    words = [item.rsplit(':', 1)[0] for item in items]
    word_labels = [item.rsplit(':', 1)[1] for item in items]
    return {
        'intent_label': intent_label,
        'words': " ".join(words),
        'words_label': " ".join(word_labels),
    }


def create_sentence_tag(data):
    sentences = []
    tags = []
    intent_tags = []
    alltags = []
    for i in range(data.shape[0]):
        sentence = data.iloc[i]['words'].split()
        sentence_tag = data.iloc[i]['words_label'].split()
        intent_tag = data.iloc[i]['intent_label']
        sentences.append(sentence)
        tags.append(sentence_tag)
        intent_tags.append(intent_tag)
        alltags += sentence_tag
    print("<<<--- Train --->>>")
    print(len(sentences), len(tags), len(intent_tags))
    alltags = list(set(alltags))
    allintents = list(set(intent_tags))
    return sentences, tags, intent_tags, alltags, allintents


def create_tokens_and_labels(id, sample):
    intent = sample['intent']
    utt = sample['utt']
    annot_utt = sample['annot_utt']
    tokens = utt.split()
    labels = []
    label = 'O'
    split_annot_utt = annot_utt.split()
    idx = 0
    BIO_SLOT = False
    while idx < len(split_annot_utt):
        if split_annot_utt[idx].startswith('['):
            label = split_annot_utt[idx].lstrip('[')
            idx += 2
            BIO_SLOT = True
        elif split_annot_utt[idx].endswith(']'):
            if split_annot_utt[idx-1] ==":":
                labels.append("B-"+label)
                label = 'O'
                idx += 1
            else:
                labels.append("I-"+label)
                label = 'O'
                idx += 1
            BIO_SLOT = False
        else:
            if split_annot_utt[idx-1] ==":":
                labels.append("B-"+label)
                idx += 1
            elif BIO_SLOT == True:
                labels.append("I-"+label)
                idx += 1
            else:
                labels.append("O")
                idx += 1                  

    if len(tokens) != len(labels):
        raise ValueError(f"Len of tokens, {tokens}, doesnt match len of labels, {labels}, "
                          f"for id {id} and annot utt: {annot_utt}")
    return tokens, labels, intent


def Read_Massive_dataset(massive_raw):
    sentences_tr, tags_tr, intent_tags_tr = [], [], []
    sentences_val, tags_val, intent_tags_val = [], [], []
    sentences_test, tags_test, intent_tags_test = [], [], []
    all_tags, all_intents = [], []

    for id, sample in enumerate(massive_raw):
        if sample['partition'] == 'train':
            tokens, labels, intent = create_tokens_and_labels(id, sample)
            sentences_tr.append(tokens)
            tags_tr.append(labels)
            intent_tags_tr.append(intent)
            all_tags += labels

        if sample['partition'] == 'dev':
            tokens, labels, intent = create_tokens_and_labels(id, sample)
            sentences_val.append(tokens)
            tags_val.append(labels)
            intent_tags_val.append(intent)
            all_tags += labels

        if sample['partition'] == 'test':
            tokens, labels, intent = create_tokens_and_labels(id, sample)
            sentences_test.append(tokens)
            tags_test.append(labels)
            intent_tags_test.append(intent)
            all_tags += labels
        

    all_tags = list(set(all_tags))


    allintents = intent_tags_tr + intent_tags_val + intent_tags_test
    all_intents = list(set(allintents))
    return sentences_tr, tags_tr, intent_tags_tr, sentences_val, tags_val, intent_tags_val, sentences_test, tags_test, intent_tags_test, all_tags, all_intents


def parse_excel_data(dir, save_dir=None):
    data_dir = dir + '\\Data.xlsx'
    data = pd.read_excel(data_dir)
    df = pd.DataFrame(data)

    tokens = df.iloc[::3].values.tolist()
    tags = df.iloc[1::3].values.tolist()
    queries = df.iloc[2::3].values.tolist()
    data = []

    for i in range(len(queries)):
        domain = queries[i][1]
        intent = queries[i][2]
        domain_intent_info = [domain, intent]

        token = [x for x in tokens[i] if str(x) != 'nan']
        tag = [x for x in tags[i] if str(x) != 'nan']
        tagged_slots = list(zip(token, tag))
        print([domain_intent_info, tagged_slots])
        data.append([domain_intent_info, tagged_slots])

    return data


def get_token_slots(temp_tokens, defined_slots):
    token = []
    tag = []
    for ele in temp_tokens:
        str_token = str(ele)
        if str(ele) == "nan":
            break

        if ele in defined_slots.keys():
            possible = random.choice(defined_slots[ele]).split(' ')
            token = token + possible
            tag.append(ele)
            if len(possible) > 1:
                tag = tag + ['I-' + ele.split('-')[1]] * (len(possible) - 1)
        else:
            token.append(ele)
            tag.append('O')

    return token, tag


def generate_data_templates(dir, repeat, save_dir=None):
    data_dir = dir + '\\Templates.xlsx'

    with open(dir+'\\possible_slots.json', encoding='utf-8') as json_file:
        defined_slots = json.load(json_file)

    data = pd.read_excel(data_dir)
    df = pd.DataFrame(data)

    data_train = []
    data_test = []
    tags = df.iloc[1::2].values.tolist()
    tokens = df.iloc[::2].values.tolist()
    tags = tags + [['آب و هوا', 'پرسش']]

    tokens_train, tokens_test, tags_train, tags_test = train_test_split(tokens, tags, test_size=0.2, random_state=44)

    for i in range(len(tags_train)):
        domain = tags_train[i][0]
        intent = tags_train[i][1]
        domain_intent_info = [domain, intent]

        for j in range(repeat):
            token, tag = get_token_slots(tokens_train[i], defined_slots)
            tagged_slots = list(zip(token, tag))
            data_train.append([domain_intent_info, tagged_slots])
            # print([domain_intent_info, tagged_slots])

    for i in range(len(tags_test)):
        domain = tags_test[i][0]
        intent = tags_test[i][1]
        domain_intent_info = [domain, intent]

        token, tag = get_token_slots(tokens_test[i], defined_slots)
        tagged_slots = list(zip(token, tag))
        data_test.append([domain_intent_info, tagged_slots])


    return data_train, data_test


def write_slots(arr, dir):
    with open(dir, 'w', encoding='utf-8') as file:
        for elem in arr:
            domain = elem[0][0]
            intent = elem[0][1]

            if len(elem) != 2:
                continue
            for token, slot in elem[1]:
                file.write(str(token) + ':' + str(slot) + '\t')

            file.write('<=>\t' + intent + '\t' + domain + '\n')


def write_vocab(vocab, dir):
    with open(dir, 'w', encoding='utf-8') as file:
        for word in vocab:
            file.write(word + '\n')


def write_data(data,temp_train, temp_test, dir):
    intents = set()
    domains = set()
    slots = set()

    for elem in data:
        if len(elem) != 2:
            continue
        # print(elem[0])
        domains.add(elem[0][0].strip())
        intents.add(elem[0][1].strip())

        for token, slot in elem[1]:
            slots.add(slot.strip())

    train_val, test = train_test_split(data, test_size=0.15, random_state=42, shuffle=True)
    train, val = train_test_split(train_val, test_size=0.15, random_state=42, shuffle=True)
    test = test + temp_test
    train = train + temp_train

    print("Train size:", len(train))
    print("Test size:", len(test))

    write_slots(train, dir+'train')
    write_slots(test, dir+'test')
    write_slots(val, dir+'valid')

    write_vocab(intents, dir+'vocab.intent')
    write_vocab(domains, dir+'vocab.domain')
    write_vocab(slots, dir+'vocab.slot')


def parse_ourData_newformat(dir, save_dir=None):

    ## Downloading dataset from github
    # for filename in ["train", "valid", "test", "vocab.intent", "vocab.slot"]:
    #     path = Path(filename)
    #     if not path.exists():
    #         print(f"Downloading {filename}...")
    #         urlretrieve(dir + filename + "?raw=true", path)


    ##  Reading Massive dataset

    # path = Path('amazon-massive-dataset-1.0.tar.gz')
    # if not path.exists():
    #     print("Downloading amazon-massive-dataset-1.0.tar.gz...")
    #     urlretrieve(dir + 'amazon-massive-dataset-1.0.tar.gz' + "?raw=true", path)

    # my_tar = tarfile.open(Path('amazon-massive-dataset-1.0.tar.gz'))
    # my_tar.extract("")
    # my_tar.close()
    
    # returns JSON object as 
    # a dictionary
    # massive_raw_en = []
    # with open('/content/1.0/data/en-US.jsonl', 'r') as f:
    #     for line in f:
    #         massive_raw_en.append(json.loads(line))
    #
    # # Closing file
    # f.close()
    #
    # massive_raw_fa = []
    # with open('/content/1.0/data/en-US.jsonl', 'r') as f:
    #     for line in f:
    #         massive_raw_fa.append(json.loads(line))
    #
    # # Closing file
    # f.close()

    # sentences_tr, tags_tr, intent_tags_tr, sentences_val, tags_val, intent_tags_val, sentences_test, tags_test, intent_tags_test, all_tags, all_intents = Read_Massive_dataset(massive_raw_en)
    # sentences_tr_pr, tags_tr_pr, intent_tags_tr_pr, sentences_val_pr, tags_val_pr, intent_tags_val_pr, sentences_test_pr, tags_test_pr, intent_tags_test_pr, all_tags_pr, all_intents_pr = Read_Massive_dataset(massive_raw_fa)
    # Data={}
    # Data["tr_inputs"], Data["tr_tags"], Data["tr_intents"] = sentences_tr, tags_tr, intent_tags_tr
    # Data["val_inputs"], Data["val_tags"], Data["val_intents"] = sentences_test_pr, tags_test_pr, intent_tags_test_pr
    # Data["test_inputs"], Data["test_tags"], Data["test_intents"] = sentences_val_pr, tags_val_pr, intent_tags_val_pr


    lines_train = Path(dir + 'train').read_text('utf-8').strip().splitlines()
    lines_validation = Path(dir + 'valid').read_text('utf-8').strip().splitlines()
    lines_test = Path(dir + 'test').read_text('utf-8').strip().splitlines()


    parsed = [parse_line(line) for line in lines_train]
    df_train = pd.DataFrame([p for p in parsed if p is not None])
    df_validation = pd.DataFrame([parse_line(line) for line in lines_validation])
    df_test = pd.DataFrame([parse_line(line) for line in lines_test])


    Data = {}
    Data["tr_inputs"], Data["tr_tags"], Data["tr_intents"], tr_alltags, tr_allintents = create_sentence_tag(df_train)
    Data["val_inputs"], Data["val_tags"], Data["val_intents"], val_alltags, val_allintents = create_sentence_tag(df_validation)
    Data["test_inputs"], Data["test_tags"], Data["test_intents"], test_alltags, test_allintents = create_sentence_tag(df_test)


    
    Data["tr_tokens"] = Data["tr_inputs"]
    Data["val_tokens"] = Data["val_inputs"]
    Data["test_tokens"] = Data["test_inputs"]

    alltags = tr_alltags + val_alltags + test_alltags
    allintents = tr_allintents + val_allintents + test_allintents

    print(set(alltags))
    print(set(allintents))

    alltags = list(set(alltags))
    allintents = list(set(allintents))
    
    dict2 = {}
    dict_rev2 = {}
    inte2 = {}
    inte_rev2 = {}

    for i, tag in enumerate(alltags):
        dict_rev2[tag] = i + 1
        dict2[i + 1] = tag
    print("Slots labels: ", alltags)
    print("Number of Slots labels :", len(alltags))
    # del alltags

    for i, tag in enumerate(allintents):
        inte_rev2[tag] = i + 1
        inte2[i + 1] = tag
    print("Intent labels: ", allintents)
    print("Number of Intents labels :", len(allintents))
    # del allintents
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if save_dir is not None:
        save_obj(Data, save_dir + '/Data')
        save_obj(dict2, save_dir + '/dict2')
        save_obj(dict_rev2, save_dir + '/dict_rev2')
        save_obj(inte2, save_dir + '/inte2')
        save_obj(inte_rev2, save_dir + '/inte_rev2')
        save_obj(alltags, save_dir + '/alltags')


