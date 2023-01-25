from utils import read_train_and_val_data, convert_data_to_nli_format, tokenize_data, MyDataset, read_test_data, evaluate_val_predictions, aggregate_predictions, aggregate_test_predictions
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import torch

#
# arguments_test, labels_test = read_test_data()
# test = convert_data_to_nli_format(arguments_test, labels_test, definition="description")
# print(test)

# arguments_train_df, arguments_val_df, level2_labels_train_df, level2_labels_val_df = read_train_and_val_data()
# train = convert_data_to_nli_format(arguments_train_df, level2_labels_train_df, definition="description")
# val = convert_data_to_nli_format(arguments_val_df, level2_labels_val_df, definition="description")
#
# # print(level2_labels_val_df)
#
# #use samples
# train = train.sample(n=1000, random_state=42)
# argument_sample = arguments_val_df['Argument ID'].sample(n = 100, random_state=42).tolist()
# val_sample = val.loc[val['Argument ID'].isin(argument_sample)].groupby(['Entailment', 'Level2_Value']).sample(n=50, random_state=42, replace=True)
# # print(val)
# print(argument_sample)
# print(val_sample)
# print(val_sample.loc[(val_sample['Argument ID'] == "A05079") & (val_sample['Level2_Value'] == "Achievement")])

#
# preds_df = pd.DataFrame({'Argument ID': val['Argument ID'],
#                          'Level2_Value': val['Level2_Value'],
#                          'Prediction': np.random.randint(0, 2, val.shape[0])})
#
# print(preds_df)

# f1_dict = evaluate_val_predictions(preds_df, level2_labels_val_df, threshold=0.5)
# print(f1_dict)
# thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# f1_across_thresholds = [evaluate_val_predictions(preds_df, level2_labels_val_df, threshold) for threshold in thresholds]
# max_index = f1_across_thresholds.index(max(f1_across_thresholds))
# print(f1_across_thresholds)
# print(max_index)
#
# aggregated_preds = aggregate_test_predictions(preds_df, threshold=0.5)
# aggregated_preds.to_csv('test.tsv', header=True, index=False, sep='\t')

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# train_encodings, train_labels, max_length = tokenize_data(train, tokenizer, max_length=None)
# val_encodings, val_labels = tokenize_data(val, tokenizer, max_length=max_length)
# # print(max_length)
# # print(train_tokenized["input_ids"][0])
# # print(val_tokenized["input_ids"][0])
#
# train_dataset = MyDataset(train_encodings, train_labels)
# val_dataset = MyDataset(val_encodings, val_labels)

#
# arguments_test_df, labels_test_df = read_test_data()
# test = convert_data_to_nli_format(arguments_test_df, labels_test_df, definition="description")
#
# test_argument_sample = arguments_test_df['Argument ID'].sample(n=100, random_state=42).tolist()
# test = test.loc[test['Argument ID'].isin(test_argument_sample)].groupby(['Level2_Value']).sample(n=100,
#                                                                                                  random_state=42,
#                                                                                                  replace=True)
# print(test)

tensor = torch.tensor([0, 0, 0, 0, 0])
print(tensor)
tensor_counts = torch.bincount(tensor)
tensor_weights = torch.flip(tensor_counts, [0])
print(tensor_weights.float())
print(tensor_counts.size(dim=0))