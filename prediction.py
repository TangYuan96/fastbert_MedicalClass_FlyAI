# -*- coding: utf-8 -*
from flyai.framework import FlyAI

import os, sys
fastbert_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(fastbert_dir)
import torch
from fastbert import FastBERT
from getDataSet import editDFLine, getModelData

model_saving_path = "./data/output/model/best.bin"

class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''

        self.dfTrainLabelList, self.dfTrainSList, self.dfVaLabelList, self.dfVaSList, self.labels, self.Label2Char= getModelData(EXMode=1)

        self.model = FastBERT(
            kernel_name="uer_bert_tiny_zh",
            labels=self.labels,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.model.load_model(model_saving_path)
        pass

    def predict(self, title, text):
        '''
        模型预测返回结果
        :param input:  评估传入样例 {"title":"文本","text":"文本"}
        :return: 模型预测成功之后返回给系统样例 {"label":"文本"}
        '''

        labelNum, exec_layer= self.model(title+text)

        labelText =self.Label2Char[labelNum]

        return {"label": labelText}





if __name__ == '__main__':
    predict1 = Prediction()
    predict1.load_model()
    label = predict1.predict("肩膀脖子后背酸痛是怎么回事?","我的工作主要是在电脑前,以前也有肩膀不舒服的感觉")
    print("label:",label)
