import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.metrics import classification_report

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 0)


def read_train_and_val_data(long_format=True):
    arguments_train_df = pd.read_csv('../../data/raw/touche23/arguments-training.tsv', sep='\t')
    arguments_val_df = pd.read_csv('../../data/raw/touche23/arguments-validation.tsv', sep='\t')
    level2_labels_train_df = pd.read_csv('../../data/raw/touche23/labels-training.tsv', sep='\t')
    level2_labels_val_df = pd.read_csv('../../data/raw/touche23/labels-validation.tsv', sep='\t')

    if long_format:
        level2_values = list(level2_labels_train_df.columns)[1:]
        level2_labels_train_df = pd.melt(level2_labels_train_df, id_vars="Argument ID", value_vars=level2_values,
                                         var_name='Level2_Value', value_name="Entailment")
        level2_labels_val_df = pd.melt(level2_labels_val_df, id_vars="Argument ID", value_vars=level2_values,
                                       var_name='Level2_Value', value_name="Entailment")

    return arguments_train_df, arguments_val_df, level2_labels_train_df, level2_labels_val_df


def read_test_data(which="test", long_format=True):
    if which == "test":
        arguments_test_df = pd.read_csv('../../data/raw/touche23/arguments-test.tsv', sep='\t')
    if which == "nahjalbalagha":
        arguments_test_df = pd.read_csv('../../data/raw/touche23/arguments-test-nahjalbalagha.tsv',
                                        sep='\t')

    level2_values = pd.read_csv('../../data/raw/touche23/labels-validation.tsv', sep='\t').columns[1:]
    level2_labels_test_df = arguments_test_df[["Argument ID"]]
    for value in level2_values:
        level2_labels_test_df[value] = 0

    if long_format:
        level2_values = list(level2_labels_test_df.columns)[1:]
        level2_labels_test_df = pd.melt(level2_labels_test_df, id_vars="Argument ID", value_vars=level2_values,
                                        var_name='Level2_Value', value_name="Entailment")

    return arguments_test_df, level2_labels_test_df


def convert_data_to_nli_format(arguments, labels, definition=None):
    arguments = arguments[["Argument ID", "Premise"]]
    df = pd.merge(arguments, labels, how="left", on="Argument ID")

    if definition is None:
        return df

    if definition == "description":
        definition_df = pd.read_csv('../../data/raw/SemEval_ValueDescription.tsv', sep='\t')
        definition_df = definition_df[["Level2_Value", "Hypothesis"]]

    if definition == "survey":
        definition_df = pd.read_csv('../../data/raw/SemEval_QuestionnaireItems.tsv', sep='\t')
        definition_df = definition_df[["Level2_Value", "Hypothesis"]]

    df = pd.merge(df, definition_df, how="left", on="Level2_Value")
    return df


def tokenize_data(df, tokenizer, max_length=None):
    tuples = list(zip(df['Premise'], df['Hypothesis'], df['Entailment']))
    input_data = []
    labels = []
    for (premise, hypothesis, label) in tuples:
        input_data.append([premise.lower(), hypothesis.lower()])
        labels.append(label)

    if max_length is None:
        tokenized_data = tokenizer(input_data, padding=True, truncation=True)
        return tokenized_data, torch.as_tensor(labels), np.array(tokenized_data['input_ids']).shape[1]
    else:
        return tokenizer(input_data, max_length=max_length, padding='max_length'), torch.as_tensor(labels)


class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def evaluate_val_predictions(predictions_df, val_df, threshold):
    # predictions_df contains three columns: Argument ID, Level2_Value, Prediction
    level2_values = val_df['Level2_Value'].unique()
    f1_ls = []
    for value in level2_values:
        df_temp = predictions_df.loc[predictions_df['Level2_Value'] == value]
        preds_df = aggregate_predictions(df_temp, threshold)
        preds_df = pd.merge(preds_df, val_df.loc[val_df['Level2_Value'] == value], how='left', on='Argument ID')
        classification_results = classification_report(preds_df['Entailment'].tolist(),
                                                       preds_df['Prediction'].tolist(),
                                                       output_dict=True)
        f1 = classification_results['1']['f1-score']
        f1_ls.append(f1)

    return sum(f1_ls) / len(f1_ls)

def aggregate_predictions(df, threshold):
    # the df here contains only predictions for a specific level2 value
    preds = df.groupby('Argument ID')['Prediction'].mean() > threshold
    preds_df = preds.to_frame()
    preds_df = preds_df.reset_index()
    preds_df['Prediction'] = preds_df['Prediction'].astype(int)

    return preds_df

def aggregate_test_predictions(test_predictions_df, threshold):
    # the test_predictions_df contains three columns: Argument ID, Level2_Value, Prediction for all level2 values.
    preds = test_predictions_df.groupby(['Argument ID', 'Level2_Value'])['Prediction'].mean() > threshold
    preds_df = preds.to_frame()
    preds_df = preds_df.reset_index()
    preds_df['Prediction'] = preds_df['Prediction'].astype(int)

    preds_df_wide = preds_df.pivot(index='Argument ID', columns='Level2_Value', values='Prediction')
    return(preds_df_wide)
