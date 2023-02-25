import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration

pd.set_option('display.max_rows', 20)
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


def convert_data_to_nli_format(arguments, labels, definition, paraphrases="no"):
    arguments = arguments[["Argument ID", "Premise"]]

    if paraphrases == "yes":
        paraphrases_df = pd.read_csv('../../data/raw/paraphrases.tsv', sep='\t')
        arguments = pd.concat([arguments, paraphrases_df])

    df = pd.merge(arguments, labels, how="left", on="Argument ID")

    if definition == "none":
        df['Hypothesis'] = df['Level2_Value']
        return df

    if definition == "description":
        definition_df = pd.read_csv('../../data/raw/SemEval_ValueDescription.tsv', sep='\t')
        definition_df = definition_df[["Level2_Value", "Hypothesis"]]

    if definition == "survey":
        definition_df = pd.read_csv('../../data/raw/SemEval_QuestionnaireItems.tsv', sep='\t')
        definition_df = definition_df[["Level2_Value", "Hypothesis"]]

    if definition == "both":
        definition_df1 = pd.read_csv('../../data/raw/SemEval_ValueDescription.tsv', sep='\t')
        definition_df1 = definition_df1[["Level2_Value", "Hypothesis"]]

        definition_df2 = pd.read_csv('../../data/raw/SemEval_QuestionnaireItems.tsv', sep='\t')
        definition_df2 = definition_df2[["Level2_Value", "Hypothesis"]]

        definition_df = pd.concat([definition_df1, definition_df2])

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
        return tokenizer(input_data, max_length=max_length, padding='max_length', truncation=True), torch.as_tensor(labels)


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

def t5_vamsi_paraphraser(arguments_df, argument_ids): #do this only for the train set
    # values for which to be paraphrased: Stimulation, Hedonism, Power: dominance, Face, Conformity: interpersonal, Humility
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

    arguments = arguments_df.loc[arguments_df['Argument ID'].isin(argument_ids)]['Premise'].tolist()
    # print(arguments)
    paraphrase_ls = []
    for argument in arguments:
        inputs = tokenizer.encode_plus("paraphrase: " + argument + " </s>",
                                        pad_to_max_length=True,
                                        return_tensors="pt")

        outputs = model.generate(input_ids=inputs["input_ids"],
                                 attention_mask=inputs["attention_mask"],
                                 max_length=256,
                                 do_sample=True,
                                 top_k=120,
                                 top_p=0.75,
                                 early_stopping=True,
                                 num_return_sequences=3)

        paraphrases = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        paraphrase_ls = paraphrase_ls + paraphrases

    paraphrases_df = pd.DataFrame({"Argument ID": np.repeat(argument_ids, 3),
                                   "Premise": paraphrase_ls})
    paraphrases_df.to_csv('../../data/raw/paraphrases.tsv', sep='\t', index=False)

# def t5_paraphraser(arguments_df, argument_ids): #do this only for the train set
#     # values for which to be paraphrased: Stimulation, Hedonism, Power: dominance, Face, Conformity: interpersonal, Humility
#     tokenizer = T5Tokenizer.from_pretrained("t5-small")
#     model = T5ForConditionalGeneration.from_pretrained("t5-small")
#
#     arguments = arguments_df.loc[arguments_df['Argument ID'].isin(argument_ids)]['Premise'].tolist()
#     print(arguments)
#
#     print(tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True))