import pandas as pd
from torch.utils.data import Dataset
import torch

def readSemEvalData(long_format=True):
    # import arguments and labels
    argumentsTrainDf = pd.read_csv('../../data/raw/touche23/arguments-training.tsv', sep='\t')
    argumentsTrainDf['Split'] = "train"
    argumentsValDf = pd.read_csv('../../data/raw/touche23/arguments-validation.tsv', sep='\t')
    argumentsValDf['Split'] = "val"
    argumentsTestDf = pd.read_csv('../../data/raw/touche23/arguments-test.tsv', sep='\t')

    argumentsDf = pd.concat([argumentsTrainDf, argumentsValDf], ignore_index=True)

    # level 2 labels
    level2LabelsTrainDf = pd.read_csv('../../data/raw/touche23/labels-training.tsv', sep='\t')
    level2LabelsValDf = pd.read_csv('../../data/raw/touche23/labels-validation.tsv', sep='\t')
    level2LabelsDf = pd.concat([level2LabelsTrainDf, level2LabelsValDf], ignore_index=True)

    # level 1 labels
    level1LabelsTrainDf = pd.read_csv('../../data/raw/touche23/level1-labels-training.tsv', sep='\t')
    level1LabelsValDf = pd.read_csv('../../data/raw/touche23/level1-labels-validation.tsv', sep='\t')
    level1LabelsDf = pd.concat([level1LabelsTrainDf, level1LabelsValDf], ignore_index=True)

    if long_format:
        level2Values = list(level2LabelsDf.columns)[1:]
        level2LabelsDf = pd.melt(level2LabelsDf, id_vars="Argument ID", value_vars=level2Values,
                                 var_name='Level2 Hypothesis', value_name="Entailment")
        level1Values = list(level1LabelsDf.columns)[1:]
        level1LabelsDf = pd.melt(level1LabelsDf, id_vars="Argument ID", value_vars=level1Values,
                                 var_name='Level1 Hypothesis', value_name="Entailment")
        argumentsTestDf = pd.concat([argumentsTestDf.assign(Hypothesis=i) for i in level2Values], ignore_index=True)
        argumentsTestDf.rename(columns={'Hypothesis': 'Level2 Hypothesis'}, inplace=True)

    return argumentsDf, level2LabelsDf, level1LabelsDf, argumentsTestDf

def convertSemEvalDataToTerFormat(arguments, labels=None, value_level="2", test=False):
    #for the arguments df, we need only id and premises

    if test:
        df = arguments[["Argument ID", "Premise", "Level2 Hypothesis"]]
        return df
    else:
        arguments = arguments[["Argument ID", "Premise", "Split"]]
        df = pd.merge(arguments, labels, how="left", on="Argument ID")
        finalColumnsToKeep = ['Argument ID', 'Premise', 'Hypothesis', 'Entailment']

    if value_level == "2":
        df.rename(columns={'Level2 Hypothesis': 'Hypothesis'}, inplace=True)

    if value_level == "1":
        df.rename(columns={'Level1 Hypothesis': 'Hypothesis'}, inplace=True)

    if value_level == "0":
        dictionaryDfLevel1 = pd.read_csv('../../data/raw/touche23/semEvalDictionary - level1.tsv', sep='\t')
        dictionaryDfLevel0 = pd.read_csv('../../data/raw/touche23/semEvalDictionary - level0.tsv', sep='\t')

        df = pd.merge(df, dictionaryDfLevel1, how="left", on="Level2 Hypothesis")
        df = pd.merge(df, dictionaryDfLevel0, how="left", on=["Level1 ID", "Level1 Hypothesis"])
        df.rename(columns={'Level0 Hypothesis': 'Hypothesis'}, inplace=True)

    trainDf = df.loc[df['Split'] == "train"][finalColumnsToKeep]
    valDf = df.loc[df['Split'] == "val"][finalColumnsToKeep]
    return trainDf, valDf

def tokenize_data(df, tokenizer, test=False):
    if test:
        df["Entailment"] = 0

    tuples = list(zip(df['Premise'], df['Hypothesis'], df['Entailment']))

    inputData = []
    labels = []
    for (premise, hypothesis, label) in tuples:
        inputData.append([premise.lower(), hypothesis.lower()])
        labels.append(label)

    return tokenizer(inputData, max_length=190, padding='max_length'), torch.as_tensor(labels)

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