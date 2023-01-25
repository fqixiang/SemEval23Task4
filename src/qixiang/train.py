import pandas as pd
from utils import read_train_and_val_data, convert_data_to_nli_format, tokenize_data, MyDataset, read_test_data, \
    evaluate_val_predictions, aggregate_predictions, aggregate_test_predictions
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, \
    EarlyStoppingCallback
import argparse
import torch
from torch import nn
import os
import math
import numpy as np

torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        type=int,
                        default=None)
    parser.add_argument("--gradient_step_size",
                        type=int,
                        default=1)
    parser.add_argument("--definition",
                        type=str,
                        default="description")
    parser.add_argument("--weighted_loss",
                        type=str,
                        default="not_weighted")
    parser.add_argument("--device",
                        type=str,
                        default=None)
    parser.add_argument("--test_mode",
                        type=str,
                        default="yes")

    args = parser.parse_args()

    batch_size = args.batch_size
    gradient_step_size = args.gradient_step_size
    device = args.device
    test_mode = args.test_mode
    definition = args.definition
    weighted_loss = args.weighted_loss

    if batch_size is None:
        return

    # define weighted loss
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            label_counts = torch.bincount(labels)

            if label_counts.size(dim=0) == 1:
                label_weights = None
            else:
                label_weights = torch.flip(label_counts, [0]).float()

            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(weight=label_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # read training data
    arguments_train_df, arguments_val_df, level2_labels_train_df, level2_labels_val_df = read_train_and_val_data()
    train = convert_data_to_nli_format(arguments_train_df, level2_labels_train_df, definition=definition)
    val = convert_data_to_nli_format(arguments_val_df, level2_labels_val_df, definition=definition)

    # get a subsample of the training data for test run
    if test_mode == "yes":
        train_argument_sample = arguments_train_df['Argument ID'].sample(n = 100, random_state=42).tolist()
        val_argument_sample = arguments_val_df['Argument ID'].sample(n=100, random_state=42).tolist()
        train = train.loc[train['Argument ID'].isin(train_argument_sample)].groupby(['Entailment', 'Level2_Value']).sample(n=50, random_state=42, replace=True)
        val = val.loc[val['Argument ID'].isin(val_argument_sample)].groupby(['Entailment', 'Level2_Value']).sample(n=50, random_state=42, replace=True)
        eval_batch_size = 128
    else:
        eval_batch_size = 1024

    # print(train)
    # print(val)
    print("Train and test sets created.")

    # tokenize the datasets and prepare the data in Dataset class
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_encodings, train_labels, max_length = tokenize_data(train, tokenizer, max_length=None)
    print("Tokenization of training data: complete.")
    val_encodings, val_labels = tokenize_data(val, tokenizer, max_length=max_length)
    print("Tokenization of validation data: complete.")

    train_dataset = MyDataset(train_encodings, train_labels)
    val_dataset = MyDataset(val_encodings, val_labels)

    # # train BERT
    if device == "hpc":
        output_dir = '/hpc/uu_cs_nlpsoc/data/qixiang/proj_semeval23_task4/models/' + definition + '_' + weighted_loss + '/'
    else:
        output_dir = '../../results/output/models/' + definition + '_' + weighted_loss + '/'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if test_mode == "yes":
        n_steps = 5
    else:
        n_steps = 2500

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=250,
        weight_decay=0.01,
        # logging_dir='./logs',
        logging_strategy='steps',
        logging_steps=n_steps,
        # metric_for_best_model='f1',
        evaluation_strategy='steps',
        eval_steps=n_steps,
        eval_delay=math.ceil(train.shape[0] / batch_size),
        save_strategy='steps',
        save_steps=n_steps,
        gradient_accumulation_steps=gradient_step_size,
        load_best_model_at_end=True,
        save_total_limit=5,
        seed=42,
        data_seed=42
    )

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                  num_labels=2)

    if weighted_loss == "not_weighted":
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        )
    else:
        trainer = CustomTrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        )

    trainer.train()

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

    # make predictions on the test set
    arguments_test_df, labels_test_df = read_test_data(which="test")
    test = convert_data_to_nli_format(arguments_test_df, labels_test_df, definition=definition)

    if test_mode == "yes":
        test_argument_sample = arguments_test_df['Argument ID'].sample(n = 100, random_state=42).tolist()
        test = test.loc[test['Argument ID'].isin(test_argument_sample)].groupby(['Level2_Value']).sample(n=100, random_state=42, replace=True)

    test_encodings, test_labels = tokenize_data(test, tokenizer, max_length=max_length)
    print("Tokenization of test data: complete.")

    test_dataset = MyDataset(test_encodings, test_labels)
    preds_test = np.argmax(trainer.predict(test_dataset)[0], axis=1)
    preds_test_df = pd.DataFrame({'Argument ID': test['Argument ID'],
                                  'Level2_Value': test['Level2_Value'],
                                  'Prediction': preds_test})

    preds_test_df.to_csv(output_dir+'test_predictions_raw.tsv', index=False, header=True, sep='\t')

    # aggregate the predictions on the test set and save them to tsv
    aggregated_test_preds_df = aggregate_test_predictions(preds_test_df, threshold=thresholds[max_index])
    aggregated_test_preds_df.to_csv(output_dir+'test_predictions_agg.tsv', header=True, index=True, sep='\t')

    print("Best threshold value: " + str(thresholds[max_index]))
    print("Best f1 score is: " + str(max(f1_across_thresholds)))


if __name__ == '__main__':
    main()
