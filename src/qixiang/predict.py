from utils import read_test_data, convert_data_to_nli_format, tokenize_data, MyDataset, read_train_and_val_data, \
    evaluate_val_predictions, aggregate_test_predictions
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--definition",
                        type=str,
                        default=None)
    parser.add_argument("--weighted_loss",
                        type=str,
                        default="not_weighted")
    parser.add_argument("--device",
                        type=str,
                        default=None)
    parser.add_argument("--paraphrases",
                        type=str,
                        default="no")
    parser.add_argument("--model_number",
                        type=str,
                        default=None)
    parser.add_argument("--test_mode",
                        type=str,
                        default="yes")

    args = parser.parse_args()

    model_number = args.model_number
    definition = args.definition
    test_mode = args.test_mode
    weighted_loss = args.weighted_loss
    device = args.device
    paraphrases = args.paraphrases

    # read train, val and test data
    arguments_train_df, arguments_val_df, level2_labels_train_df, level2_labels_val_df = read_train_and_val_data()
    arguments_test1_df, labels_test1_df = read_test_data(which="test")
    arguments_test2_df, labels_test2_df = read_test_data(which="nahjalbalagha")

    train = convert_data_to_nli_format(arguments_train_df, level2_labels_train_df, definition=definition, paraphrases=paraphrases)
    val = convert_data_to_nli_format(arguments_val_df, level2_labels_val_df, definition=definition)
    test1 = convert_data_to_nli_format(arguments_test1_df, labels_test1_df, definition=definition)
    test2 = convert_data_to_nli_format(arguments_test2_df, labels_test2_df, definition=definition)

    # get a subsample of the training data for test run
    if test_mode == "yes":
        train_argument_sample = arguments_train_df['Argument ID'].sample(n = 100, random_state=42).tolist()
        val_argument_sample = arguments_val_df['Argument ID'].sample(n=100, random_state=42).tolist()
        train = train.loc[train['Argument ID'].isin(train_argument_sample)].groupby(['Entailment', 'Level2_Value']).sample(n=50, random_state=42, replace=True)
        val = val.loc[val['Argument ID'].isin(val_argument_sample)].groupby(['Entailment', 'Level2_Value']).sample(n=50, random_state=42, replace=True)
        test1_argument_sample = arguments_test1_df['Argument ID'].sample(n = 100, random_state=42).tolist()
        test1 = test1.loc[test1['Argument ID'].isin(test1_argument_sample)].groupby(['Level2_Value']).sample(n=100, random_state=42, replace=True)
        test2_argument_sample = arguments_test2_df['Argument ID'].sample(n = 100, random_state=42).tolist()
        test2 = test2.loc[test2['Argument ID'].isin(test2_argument_sample)].groupby(['Level2_Value']).sample(n=100, random_state=42, replace=True)
        eval_batch_size = 128
    else:
        eval_batch_size = 1024

    # tokenize the datasets and prepare the data in Dataset class
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_encodings, train_labels, max_length = tokenize_data(train, tokenizer, max_length=None)
    print("Tokenization of training data: complete.")
    print("Maximum tokenization length is: " + str(max_length))

    val_encodings, val_labels = tokenize_data(val, tokenizer, max_length=max_length)
    print("Tokenization of validation data: complete.")
    test1_encodings, test1_labels = tokenize_data(test1, tokenizer, max_length=max_length)
    test2_encodings, test2_labels = tokenize_data(test2, tokenizer, max_length=max_length)
    print("Tokenization of test data: complete.")

    val_dataset = MyDataset(val_encodings, val_labels)
    test1_dataset = MyDataset(test1_encodings, test1_labels)
    test2_dataset = MyDataset(test2_encodings, test2_labels)

    # load trained BERT
    if device == "hpc":
        output_dir = '/hpc/uu_cs_nlpsoc/data/qixiang/proj_semeval23_task4/models/' + definition + '_' + weighted_loss + '_' + paraphrases + '/'
    else:
        output_dir = '../../results/output/models/' + definition + '_' + weighted_loss + '_' + paraphrases + '/'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = AutoModelForSequenceClassification.from_pretrained(output_dir + "checkpoint-" + model_number,
                                                               num_labels=2)
    training_args = TrainingArguments(
        output_dir=output_dir,          # output directory
        per_device_eval_batch_size=eval_batch_size,
    )

    trainer = Trainer(model=model,
                      args=training_args)

    # make predictions on the validation set
    preds_val = np.argmax(trainer.predict(val_dataset)[0], axis=1)
    preds_val_df = pd.DataFrame({'Argument ID': val['Argument ID'],
                                 'Level2_Value': val['Level2_Value'],
                                 'Prediction': preds_val})
    preds_val_df.to_csv(output_dir+'val_predictions_raw.tsv', index=False, header=True, sep='\t')

    # find best threshold for val, f1 scoring function required
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    f1_across_thresholds = [evaluate_val_predictions(preds_val_df, level2_labels_val_df, threshold) for threshold in
                            thresholds]
    max_index = f1_across_thresholds.index(max(f1_across_thresholds))
    print("Best threshold value: " + str(thresholds[max_index]))
    print("Best f1 score is: " + str(max(f1_across_thresholds)))

    # make predictions on the test set
    preds_test1 = np.argmax(trainer.predict(test1_dataset)[0], axis=1)
    preds_test1_df = pd.DataFrame({'Argument ID': test1['Argument ID'],
                                   'Level2_Value': test1['Level2_Value'],
                                   'Prediction': preds_test1})

    preds_test2 = np.argmax(trainer.predict(test2_dataset)[0], axis=1)
    preds_test2_df = pd.DataFrame({'Argument ID': test2['Argument ID'],
                                   'Level2_Value': test2['Level2_Value'],
                                   'Prediction': preds_test2})

    preds_test1_df.to_csv(output_dir+'test1_predictions_raw.tsv', index=False, header=True, sep='\t')
    preds_test2_df.to_csv(output_dir + 'test2_predictions_raw.tsv', index=False, header=True, sep='\t')

    # aggregate the predictions on the test set and save them to tsv
    aggregated_test1_preds_df = aggregate_test_predictions(preds_test1_df, threshold=thresholds[max_index])
    aggregated_test1_preds_df.to_csv(output_dir+'test1_predictions_agg.tsv', header=True, index=True, sep='\t')

    aggregated_test2_preds_df = aggregate_test_predictions(preds_test2_df, threshold=thresholds[max_index])
    aggregated_test2_preds_df.to_csv(output_dir+'test2_predictions_agg.tsv', header=True, index=True, sep='\t')

    print("Best threshold value: " + str(thresholds[max_index]))
    print("Best f1 score is: " + str(max(f1_across_thresholds)))

if __name__ == '__main__':
    main()