# -*- coding: utf-8 -*- 
# author: Honay.King

import os
import json
import jieba
import numpy as np


def load_dict(dictFile):
    if not os.path.exists(dictFile):
        print('[ERROR] load_dict failed! | The params {}'.format(dictFile))
        return None
    with open(dictFile, 'r', encoding='UTF-8') as df:
        dictF = json.load(df)
    text2id, id2text = dict(), dict()
    count = 0
    for key, value in dictF.items():
        text2id[key] = count
        id2text[count] = key
        count += 1
    return text2id, id2text


def load_labeldict(dictFile):
    if not os.path.exists(dictFile):
        print('[ERROR] load_labeldict failed! | The params {}'.format(dictFile))
        return None
    with open(dictFile, 'r', encoding='UTF-8') as df:
        label2id = json.load(df)
    id2label = dict()
    for key, value in label2id.items():
        id2label[value] = key
    return label2id, id2label


def read_data(data, textdict, labeldict):
    text_data, label_data = list(), list()
    for ind, row in data.iterrows():
        text_line = jieba.lcut(row['title'] + row['text'])
        tmp_text = list()
        for text in text_line:
            if text in textdict.keys():
                tmp_text.append(textdict[text])
            else:
                tmp_text.append(textdict['_unk_'])
        text_data.append(tmp_text)
        label = np.zeros(len(labeldict), dtype=int)
        label[labeldict[row['label']]] = 1
        label_data.append(label)
    return text_data, label_data


def pred_process(title, text, textdict, max_len=68):
    text_line = jieba.lcut(title+text)
    tmp_text = list()
    for item in text_line:
        if item in textdict.keys():
            tmp_text.append(textdict[item])
        else:
            tmp_text.append(textdict['_unk_'])
        if len(tmp_text) >= max_len:
            tmp_text = tmp_text[:max_len]
        else:
            tmp_text = tmp_text + [textdict['_pad_']] * (max_len - len(tmp_text))
    return [np.array(tmp_text)]


def batch_padding(text_batch, padding, max_len=68):
    '''
    ???batch?????????????????????????????????batch???????????????????????????text_length
    ?????????
    - text_batch
    - padding: <PAD>???????????????
    '''
    batch_text = list()
    for text in text_batch:
        if len(text) >= max_len:
            batch_text.append(np.array(text[:max_len]))
        else:
            batch_text.append(np.array(text + [padding] * (max_len-len(text))))
    return batch_text


def get_batches(texts, labels, batch_size, text_padding):
    for batch_i in range(0, len(labels) // batch_size):
        start_i = batch_i * batch_size
        texts_batch = texts[start_i: start_i + batch_size]
        labels_batch = labels[start_i: start_i + batch_size]

        pad_texts_batch = batch_padding(texts_batch, text_padding)
        yield pad_texts_batch, labels_batch


def get_val_batch(texts, labels, batch_size, text_padding):
    texts_batch = texts[:batch_size]
    labels_batch = labels[:batch_size]
    pad_texts_batch = batch_padding(texts_batch, text_padding)
    return pad_texts_batch, labels_batch


if __name__ == "__main__":
    from prediction import Prediction

    model = Prediction()
    model.load_model()

    result = model.predict(title='????????????????????????????????????', text='???')
    print(result)

    exit(0)