#check simple descriptives of the big five and other related variables
#check the variance-covariance matrix

from utils import readWassa22Data, getGloveEmbeddings, wassa2022, multiRegression, training
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd

# get the train and dev sets
trainFeatures, devFeatures, trainLabels, devLabels = readWassa22Data()

# feature engineering
dfTrainGloveFeatures = getGloveEmbeddings(trainFeatures)
dfDevGloveFeatures = getGloveEmbeddings(devFeatures)

wassa2022Train = wassa2022(features=dfTrainGloveFeatures,
                           labels=trainLabels)
wassa2022Dev = wassa2022(features=dfDevGloveFeatures,
                         labels=devLabels)

# correlation and covariance matrices
dfTrainLabels = pd.DataFrame(trainLabels,
                   columns=['con', 'ope', 'ext', 'agr', 'sta'])

print(dfTrainLabels.corr())
print(dfTrainLabels.cov())

#
# # data loaders
# batchSize = dfTrainGloveFeatures.shape[0]
# trainLoader = DataLoader(dataset=wassa2022Train,
#                          batch_size=batchSize,
#                          shuffle=True)
# devLoader = DataLoader(dataset=wassa2022Dev,
#                        batch_size=batchSize,
#                        shuffle=False)
#
#
# # baseline MSE
# meanTrainLabels = [sum(subList) / len(subList) for subList in zip(*trainLabels)]
# meanTrainLabelsDict = {'con': meanTrainLabels[0],
#                        'ope': meanTrainLabels[1],
#                        'ext': meanTrainLabels[2],
#                        'agr': meanTrainLabels[3],
#                        'sta': meanTrainLabels[4]}
# mseDict = {}
# nSamples = 0
# lossFunc = nn.MSELoss(reduction='sum')
#
# for i, samples in enumerate(devLoader):
#     labels = samples['labels']
#     nSamples += samples['features'].shape[0]
#
#     for key in labels:
#         meanPredictions = torch.tensor([meanTrainLabelsDict[key]]*len(labels[key]))
#         mse = lossFunc(labels[key], meanPredictions)
#         mseDict[key] = mseDict.get(key, 0) + mse.item()
#
# mseDict = {k: v/nSamples for k, v in mseDict.items()}
#
# print(f'Baseline: {mseDict}')
# #{'con': 1.6901787018107477, 'ope': 1.410108049339223, 'ext': 3.4321973568925235, 'agr': 1.8677543568833965, 'sta': 2.884683484228972}
#
# # model training
# inputSize = 50
# outputSize = 1
# model = multiRegression(inputSize=inputSize,
#                         outputSize=outputSize)
#
# modelLosses = training(model=model,
#                        device="cpu",
#                        lrRate=0.0001,
#                        nEpochs=10,
#                        trainLoader=trainLoader)
#
# # model testing
# device = "cpu"
# lossFunc = nn.MSELoss(reduction='sum')
#
# with torch.no_grad():
#     nSamples = 0
#     mseDict = {}
#
#     for i, samples in enumerate(devLoader):
#         features = samples['features']
#         labels = samples['labels']
#         outputs = model(features)
#
#         nSamples += features.shape[0]
#
#         for key in outputs:
#             mse = lossFunc(labels[key], torch.flatten(outputs[key]))
#             mseDict[key] = mseDict.get(key, 0) + mse
#
#     mseDict = {k: v/nSamples for k, v in mseDict.items()}
#
# print(mseDict)