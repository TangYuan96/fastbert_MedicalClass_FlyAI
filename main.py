# -*- coding: utf-8 -*-
import argparse
import os

from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
import pandas as pd


from path import MODEL_PATH

'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


import os, sys
fastbert_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(fastbert_dir)
import torch
from fastbert import FastBERT
from getDataSet import editDFLine, getModelData

model_saving_path = "./data/output/model/best.bin"

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=2, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
args = parser.parse_args()


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("MedicalClass")

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''

        self.dfTrainLabelList, self.dfTrainSList, self.dfVaLabelList, self.dfVaSList, self.labels, self.Label2Char= getModelData(EXMode=1)
        pass

    def train(self, max_epoch, batch_size):
        '''
        训练模型，必须实现此方法
        :return:
        '''

        # FastBERT model
        model = FastBERT(
            kernel_name="uer_bert_tiny_zh",
            labels=self.labels,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            seq_length=145
        )

        # model.load_model("./best.bin")

        model.fit(
            self.dfTrainSList,
            self.dfTrainLabelList,
            sentences_dev=self.dfVaSList,
            labels_dev=self.dfVaLabelList,
            finetuning_epochs_num=max_epoch,
            distilling_epochs_num=0,
            report_steps=100,
            model_saving_path=model_saving_path,
            verbose=True,
            learning_rate=3e-4,
            batch_size=batch_size,
            warmup=0.5
        )
        # warmup:学习率 变化的点
        pass


if __name__ == '__main__':

    max_epoch = args.EPOCHS
    batch_size = args.BATCH

    main = Main()
    main.download_data()
    main.deal_with_data()
    main.train(max_epoch, batch_size)