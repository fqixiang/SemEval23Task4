import pandas as pd
from torch.utils.data import Dataset
import torch

def readSemEvalData(longFormat=True):
    # import arguments and labels
    argumentsDf = pd.read_csv('../../data/raw/touche23/arguments-training.tsv', sep='\t')

    # level 2 labels
    level2LabelsDf = pd.read_csv('../../data/raw/touche23/labels-training.tsv', sep='\t')

    # level 1 labels
    level1LabelsDf = pd.read_csv('../../data/raw/touche23/level1-labels-training.tsv', sep='\t')


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