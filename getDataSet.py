# -*- coding: utf-8 -*-
import pandas as pd


def editDFLine(df1):
    # 合并title 和text列->content
    df1['content'] = df1['title'] + "." + df1['text']
    # 删除 title  和text
    df1.drop(['title'], axis=1, inplace=True)
    df1.drop(['text'], axis=1, inplace=True)

    return df1

def getModelData(EXMode):
    # 1：线下训练集， 自定义比例
    # 2：线上训练集和验证集分开使用
    # 3：线上训练集和验证集汇总，自定义比例
    dataMode = EXMode

    if dataMode == 1:
        ratio = 0.97  # 训练集占的比例
        df = pd.read_csv('./data/input/MedicalClass/train.csv')
        df = editDFLine(df)
        # 取出第一列，且不重复
        lableList = df['label'].unique()

        # 将第一列的list，转成dict，方便病例转类别
        Label2Char = dict(enumerate(lableList))
        # dict 的value和key互换
        Char2Label = dict(zip(Label2Char.values(), Label2Char.keys()))
        # label换为数字
        df['label'] = df['label'].map(Char2Label)

        # 本地训练的时候分出 训练集和 验证集

        dfTrain = df.sample(n=None, frac=ratio, replace=False, random_state=123, axis=0 )
        dfVa = pd.merge(df, dfTrain, how='left', indicator=True).query(
            "_merge=='left_only'").drop('_merge', 1)

        print("df.len:",len(df))
        # print(df.head())
        print("dfTrain.len:", len(dfTrain))
        # print(dfTrain.head())
        print("dfVa.len:", len(dfVa))
        # print(dfVa.head())

        dfTrainLabelList = dfTrain['label'].values.tolist()
        dfTrainSList = dfTrain['content'].values.tolist()

        dfVaLabelList = dfVa['label'].values.tolist()
        dfVaSList = dfVa['content'].values.tolist()

    elif dataMode == 2:

        dfTrain = pd.read_csv('./data/input/MedicalClass/train.csv')
        dfTrain = editDFLine(dfTrain)

        dfVa = pd.read_csv('./data/input/MedicalClass/validation.csv')
        dfVa = editDFLine(dfVa)

        # 合成一个表，为获取所有标签
        dfAll = dfTrain.append(dfVa)
        # 取出第一列，且不重复
        lableList = dfAll['label'].unique()
        # 将第一列的list，转成dict，方便病例转类别
        Label2Char = dict(enumerate(lableList))
        # dict 的value和key互换
        Char2Label = dict(zip(Label2Char.values(), Label2Char.keys()))

        dfTrain['label'] = dfTrain['label'].map(Char2Label)
        dfVa['label'] = dfVa['label'].map(Char2Label)

        print("dfTrain.len:", len(dfTrain))
        print("dfVa.len:", len(dfVa))

        dfTrainLabelList = dfTrain['label'].values.tolist()
        dfTrainSList = dfTrain['content'].values.tolist()

        dfVaLabelList = dfVa['label'].values.tolist()
        dfVaSList = dfVa['content'].values.tolist()

    else:
        dfTrain = pd.read_csv('./data/input/MedicalClass/train.csv')
        dfVa = pd.read_csv('./data/input/MedicalClass/validation.csv')

        # 合成一个表，为获取所有标签
        df = dfTrain.append(dfVa)

        df = editDFLine(df)
        # 取出第一列，且不重复
        lableList = df['label'].unique()

        # 将第一列的list，转成dict，方便病例转类别
        Label2Char = dict(enumerate(lableList))
        # dict 的value和key互换
        Char2Label = dict(zip(Label2Char.values(), Label2Char.keys()))
        # label换为数字
        df['label'] = df['label'].map(Char2Label)

        # 本地训练的时候分出 训练集和 验证集
        ratio = 0.9  # 训练集占的比例
        dfTrain = df.sample(n=None, frac=ratio, replace=False, random_state=123, axis=0)
        dfVa = pd.merge(df, dfTrain, how='left', indicator=True).query(
            "_merge=='left_only'").drop('_merge', 1)

        print("dfTrain.len:", len(dfTrain))
        print("dfVa.len:", len(dfVa))

        dfTrainLabelList = dfTrain['label'].values.tolist()
        dfTrainSList = dfTrain['content'].values.tolist()

        dfVaLabelList = dfVa['label'].values.tolist()
        dfVaSList = dfVa['content'].values.tolist()

    labels =  [i for i in range(len(lableList))]

    return dfTrainLabelList, dfTrainSList, dfVaLabelList, dfVaSList, labels, Label2Char

if __name__ == '__main__':
    getModelData()