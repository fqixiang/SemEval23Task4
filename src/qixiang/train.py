import pandas as pd

from utils import read_train_and_val_data, convert_data_to_nli_format, tokenize_data, MyDataset, read_test_data
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, \
    EarlyStoppingCallback
import argparse
import torch
import os
import math
import numpy as np

torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        type=int,
                        default=None)
    parser.add_argument("--definition",
                        type=str,
                        default="descriptive")
    parser.add_argument("--device",
                        type=str,
                        default=None)
    parser.add_argument("--test_mode",
                        type=str,
                        default="yes")

    args = parser.parse_args()

    batch_size = args.batch_size
    device = args.device
    test_mode = args.test_mode
    definition = args.definition

    if batch_size is None:
        return

    # read training data
    arguments_train_df, arguments_val_df, level2_labels_train_df, level2_labels_val_df = read_train_and_val_data()
    train = convert_data_to_nli_format(arguments_train_df, level2_labels_train_df, definition="description")
    val = convert_data_to_nli_format(arguments_val_df, level2_labels_val_df, definition="description")

    # get a subsample of the training data for test run
    if test_mode == "yes":
        train = train.sample(n=10000, random_state=1)
        val = val.sample(n=1000, random_state=1)
        eval_batch_size = 128
    else:
        eval_batch_size = 1024

    print(train)
    print("next")
    print(val)

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
        output_dir = '/hpc/uu_cs_nlpsoc/data/qixiang/proj_semeval23_task4/models/' + definition + '/'
    else:
        output_dir = '../../results/output/models/' + definition + '/'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

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
        # gradient_accumulation_steps=math.ceil(128/batchSize),
        load_best_model_at_end=True,
        save_total_limit=5,
        seed=42,
        data_seed=42
    )

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                  num_labels=2)

    trainer = Trainer(
        model=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    trainer.train()

    # make prediction and evaluate its performance
    arguments_test, labels_test = read_test_data()
    test = convert_data_to_nli_format(arguments_test, labels_test, definition="description")
    test_encodings, test_labels = tokenize_data(test, tokenizer, max_length=max_length)
    print("Tokenization of test data: complete.")

    test_dataset = MyDataset(test_encodings, test_labels)
    preds = np.argmax(trainer.predict(test_dataset)[0], axis=1)
    preds_df = pd.DataFrame({'Argument ID': test['Argument ID'], 'Prediction': preds})

    preds_df.to_csv(output_dir+'predictions.csv', index=True, header=True)

if __name__ == '__main__':
    main()
