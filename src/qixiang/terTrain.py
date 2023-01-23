from utils import readSemEvalData, convertSemEvalDataToTerFormat, semEvalTerDataset, tokenize_data
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import argparse
import torch
import pandas as pd
import os
import math
torch.cuda.empty_cache()

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 0)

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        type=int,
                        default=None)
    parser.add_argument("--level",
                        type=str,
                        default=None)
    parser.add_argument("--device",
                        type=str,
                        default=None)
    parser.add_argument("--test_mode",
                        type=str,
                        default="yes")

    args = parser.parse_args()

    batchSize = args.batch_size
    valueLevelsForTraining = args.level
    device = args.device
    testMode = args.test_mode

    # read training data
    argumentsDf, level2LabelsDf, level1LabelsDf, _ = readSemEvalData(long_format=True)

    # convert training data to TER format
    if valueLevelsForTraining == "012":
        level0TrainDf, level0ValDf = convertSemEvalDataToTerFormat(argumentsDf, level2LabelsDf, value_level="0")
        level1TrainDf, level1ValDf = convertSemEvalDataToTerFormat(argumentsDf, level1LabelsDf, value_level="1")
        level2TrainDf, level2ValDf = convertSemEvalDataToTerFormat(argumentsDf, level2LabelsDf, value_level="2")

        trainDf = pd.concat([level0TrainDf, level1TrainDf, level2TrainDf], ignore_index=True)
        valDf = pd.concat([level0ValDf, level1ValDf, level2ValDf], ignore_index=True)

    if valueLevelsForTraining == "12":
        level1TrainDf, level1ValDf = convertSemEvalDataToTerFormat(argumentsDf, level1LabelsDf, value_level="1")
        level2TrainDf, level2ValDf = convertSemEvalDataToTerFormat(argumentsDf, level2LabelsDf, value_level="2")
        trainDf = pd.concat([level1TrainDf, level2TrainDf], ignore_index=True)
        valDf = pd.concat([level1ValDf, level2ValDf], ignore_index=True)

    if valueLevelsForTraining == "2":
        trainDf, valDf = convertSemEvalDataToTerFormat(argumentsDf, level2LabelsDf, value_level="2")

    # get a subsample of the training data for test mode
    if testMode == "yes":
        trainDf = trainDf.sample(1500, random_state=1)
        valDf = valDf.loc[valDf['Hypothesis'] == valDf['Hypothesis'].tolist()[0]]
        evalBatchSize = 128
    else:
        evalBatchSize = 1024

    print(trainDf)
    print("next")
    print(valDf)

    if batchSize is None:
        return

    # tokenize the datasets and prepare the data in Dataset class
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    trainEncodings, trainLabels = tokenize_data(trainDf, tokenizer)
    valEncodings, valLabels = tokenize_data(valDf, tokenizer)
    trainDataset = semEvalTerDataset(trainEncodings, trainLabels)
    valDataset = semEvalTerDataset(valEncodings, valLabels)

    # # train BERT
    if device == "hpc":
        outputDir = '/hpc/uu_cs_nlpsoc/data/qixiang/proj_semeval23_task4/terTrainedWithLevel' + valueLevelsForTraining + 'Values'
    else:
        outputDir = '../../results/output/models/terTrainedWithLevel' + valueLevelsForTraining + 'Values'

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    if valueLevelsForTraining == "2":
        steps = 200
    elif valueLevelsForTraining == "12":
        steps = 500
    else:
        steps = 2500

    training_args = TrainingArguments(
        output_dir=outputDir,
        num_train_epochs=5,
        per_device_train_batch_size=batchSize,
        per_device_eval_batch_size=evalBatchSize,
        warmup_steps=250,
        weight_decay=0.01,
        # logging_dir='./logs',
        logging_strategy='steps',
        logging_steps=steps,
        # metric_for_best_model='f1',
        evaluation_strategy='steps',
        eval_steps=steps,
        eval_delay=math.ceil(trainDf.shape[0]/batchSize),
        save_strategy='steps',
        save_steps=steps,
        # gradient_accumulation_steps=math.ceil(128/batchSize),
        load_best_model_at_end=True,
        save_total_limit=5,
        seed=42
    )

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        # compute_metrics=compute_metrics,
        train_dataset=trainDataset,
        eval_dataset=valDataset,
        callbacks= [EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()

if __name__ == '__main__':
    main()