# coding: utf-8
"""
An example of training single sentence classification model with
douban_book_review dataset.

@author: Weijie Liu
"""
import os, sys
fastbert_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(fastbert_dir)
import torch
from fastbert import FastBERT
from getDataSet import editDFLine, getModelData


train_dataset_path = "./data/fastbertData/data/train.tsv"
dev_dataset_path = "./data/fastbertData/data/dev.tsv"
model_saving_path = "/tmp/fastbert_douban.bin"


def loading_dataset(dataset_path):
    sents, labels = [], []
    with open(dataset_path, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if i == 0:
                continue
            line = line.strip().split('\t')
            sents.append(line[1])
            labels.append(line[0])
    return sents, labels


def main():

    # sents_train, labels_train = loading_dataset(train_dataset_path)
    # sents_dev, labels_dev = loading_dataset(dev_dataset_path)
    # labels = ["0", "1"]
    # print("Labels: ", labels)  # [0, 1]
    #
    # print("sents_train:",sents_train[0:10])
    # print("labels_train:", labels_train[0:10])
    dfTrainLabelList, dfTrainSList, dfVaLabelList, dfVaSList, labels,ia = getModelData(1)

    # FastBERT model
    model = FastBERT(
        kernel_name="uer_bert_tiny_zh",
        labels=labels,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        seq_length=150
    )

    model.fit(
        dfTrainSList,
        dfTrainLabelList,
        sentences_dev=dfVaSList,
        labels_dev=dfVaLabelList,
        finetuning_epochs_num=1,
        distilling_epochs_num=0,
        report_steps=100,
        model_saving_path=model_saving_path,
        verbose=True,
        learning_rate=2e-4,
        batch_size=8
    )


if __name__ == "__main__":
    main()