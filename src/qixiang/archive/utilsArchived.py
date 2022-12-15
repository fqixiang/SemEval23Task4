import pandas as pd
import torchtext.vocab as vocab
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch

#read data
def readWassa22Data():
    train = pd.read_csv("../data/wassa2022/messages_train_ready_for_WS.tsv", sep="\t")
    dev1 = pd.read_csv("../data/wassa2022/messages_dev_features_ready_for_WS_2022.tsv", sep="\t")
    dev2 = pd.read_csv("../data/wassa2022/goldstandard_dev_2022.tsv", sep="\t")

    #only the following variables
    varList = ["response_id", "personality_conscientiousness", "personality_openess", "personality_extraversion",
               "personality_agreeableness", "personality_stability"]

    #concatenate the essays
    train = train.groupby(varList, as_index=False).agg({'essay': ' '.join})

    #additional column names of the dev set
    namesDev2 = ["empathy", "distress", "emotion", "personality_conscientiousness", "personality_openess",
                 "personality_extraversion", "personality_agreeableness", "personality_stability",
                 "iri_perspective_taking", "iri_personal_distress", "iri_fantasy", "iri_empathatic_concern"]

    #merge the features and labels of development set
    dev2.columns = namesDev2
    dev = pd.concat([dev1, dev2], axis=1)

    #concatenate the essays of the dev set
    dev = dev.groupby(varList, as_index=False).agg({'essay': ' '.join})

    #features and labels
    trainFeatures = train['essay']
    trainLabels = train[["personality_conscientiousness", "personality_openess", "personality_extraversion",
                        "personality_agreeableness", "personality_stability"]].values.tolist()

    devFeatures = dev['essay']
    devLabels = dev[["personality_conscientiousness", "personality_openess", "personality_extraversion",
                     "personality_agreeableness", "personality_stability"]].values.tolist()

    return trainFeatures, devFeatures, trainLabels, devLabels

#use average glove embeddings as features
def getGloveEmbeddings(texts):
    # load model
    glove = vocab.GloVe(name='6B', dim=50)

    embeddings = []
    for text in texts:
        # tokenize the sentences
        text = text.translate(str.maketrans('', '', ".,!?'%"))
        text = text.lower()
        text = text.split(" ")
        text = filter(None, text)

        wordVecList = []
        for word in text:
            try:
                vector = glove.vectors[glove.stoi[word]].numpy()
                wordVecList.append(vector)
            except:
                pass

        # compute average sentence embedding
        textVec = np.mean(wordVecList, axis=0)
        embeddings.append(textVec)

    embedding_df = pd.DataFrame(data=embeddings,
                                columns=["dim%d" % (i + 1) for i in range(len(embeddings[0]))])

    return embedding_df

class wassa2022(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.n_samples = features.shape[0]

    def __getitem__(self, idx):
        con = self.labels[idx][0]
        ope = self.labels[idx][1]
        ext = self.labels[idx][2]
        agr = self.labels[idx][3]
        sta = self.labels[idx][4]

        sample = {'features': torch.tensor(self.features.iloc[idx]),
                  'labels': {'con': torch.tensor(con),
                             'ope': torch.tensor(ope),
                             'ext': torch.tensor(ext),
                             'agr': torch.tensor(agr),
                             'sta': torch.tensor(sta)}}

        return sample

    def __len__(self):
        return self.n_samples


class multiRegression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(multiRegression, self).__init__()

        self.con = nn.Linear(inputSize, outputSize)
        self.ope = nn.Linear(inputSize, outputSize)
        self.ext = nn.Linear(inputSize, outputSize)
        self.agr = nn.Linear(inputSize, outputSize)
        self.sta = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        return {
            'con': self.con(x),
            'ope': self.ope(x),
            'ext': self.ext(x),
            'agr': self.agr(x),
            'sta': self.sta(x)
        }

def criterion(lossFunc, predLabels, samples, device):
    loss = 0
    for i, key in enumerate(predLabels):
        loss += lossFunc(torch.flatten(predLabels[key]), samples['labels'][key].to(device))
    loss = loss/(len(predLabels)*samples['features'].shape[0])
    return loss

def training(model, device, lrRate, nEpochs, trainLoader):
    checkpointLosses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lrRate)
    nTotalSteps = len(trainLoader)

    lossFunc = nn.MSELoss(reduction="sum")

    for epoch in range(nEpochs):
        losses = []
        for i, samples in enumerate(trainLoader):
            features = samples['features'].to(device)

            #forward pass
            predLabels = model(features)
            loss = criterion(lossFunc, predLabels, samples, device)
            losses.append(loss.item())

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % nTotalSteps == 0:
                losses.append(loss.item())
                print(f'Epoch [{epoch+1}/{nEpochs}], Step [{i+1}/{nTotalSteps}], Loss: {loss: .4f}')

    return losses


def readSemEvalData(longFormat=True):
    # import arguments and labels
    argumentsDf = pd.read_csv('../data/touche23/arguments-training.tsv', sep='\t')

    # level 2 labels
    level2LabelsDf = pd.read_csv('../data/touche23/labels-training.tsv', sep='\t')


    # level 1 labels
    level1LabelsDf = pd.read_csv('../data/touche23/level1-labels-training.tsv', sep='\t')


    if longFormat:
        level2Values = list(level2LabelsDf.columns)[1:]
        level2LabelsDf = pd.melt(level2LabelsDf, id_vars="Argument ID", value_vars=level2Values,
                                 var_name='Level2 Hypothesis', value_name="Entailment")
        level1Values = list(level1LabelsDf.columns)[1:]
        level1LabelsDf = pd.melt(level1LabelsDf, id_vars="Argument ID", value_vars=level1Values,
                                 var_name='Level1 Hypothesis', value_name="Entailment")

    return argumentsDf, level2LabelsDf, level1LabelsDf

def convertSemEvalDataToTerFormat(arguments, labels, level):
    #for the arguments df, we need only id and premises
    arguments = arguments[["Argument ID", "Premise"]]

    df = pd.merge(arguments, labels, how="left", on="Argument ID")

    if level == "2":
        df.rename(columns={'Level2 Hypothesis': 'Hypothesis'}, inplace=True)

    if level == "1":
        df.rename(columns={'Level1 Hypothesis': 'Hypothesis'}, inplace=True)

    if level == "0":
        dictionaryDfLevel1 = pd.read_csv('../data/touche23/semEvalDictionary - level1.tsv', sep='\t')
        dictionaryDfLevel0 = pd.read_csv('../data/touche23/semEvalDictionary - level0.tsv', sep='\t')

        df = pd.merge(df, dictionaryDfLevel1, how="left", on="Level2 Hypothesis")
        df = pd.merge(df, dictionaryDfLevel0, how="left", on=["Level1 ID", "Level1 Hypothesis"])
        df.rename(columns={'Level0 Hypothesis': 'Hypothesis'}, inplace=True)

    trainIds = pd.read_csv('../data/touche23/training-Christian.tsv', sep='\t')['Argument ID'].tolist()
    testIds = pd.read_csv('../data/touche23/test-Christian.tsv', sep='\t')['Argument ID'].tolist()

    trainDf = df.loc[df['Argument ID'].isin(trainIds)][['Argument ID', 'Premise', 'Hypothesis', 'Entailment']]
    testDf = df.loc[df['Argument ID'].isin(testIds)][['Argument ID', 'Premise', 'Hypothesis', 'Entailment']]

    return trainDf, testDf

def tokenize_data(df, tokenizer):
    tuples = list(zip(df['Premise'], df['Hypothesis'], df['Entailment']))

    inputData = []
    labels = []
    for (premise, hypothesis, label) in tuples:
        inputData.append([premise.lower(), hypothesis.lower()])
        labels.append(label)
    return tokenizer(inputData, max_length=180, padding='max_length'), torch.as_tensor(labels)

class semEvalTerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)