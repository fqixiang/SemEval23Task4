from utils import readSemEvalData, convertSemEvalDataToTerFormat, semEvalTerDataset, tokenize_data
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
import argparse
import torch
import pandas as pd
import os
torch.cuda.empty_cache()

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize",
                        type=int,
                        default=None)
    parser.add_argument("--level",
                        type=str,
                        default=None)
    parser.add_argument("--device",
                        type=str,
                        default=None)
    parser.add_argument("--testing",
                        type=str,
                        default="yes")

    args = parser.parse_args()

    batchSize = args.batchSize
    level = args.level
    device = args.device
    test = args.testing

    # read training data
    argumentsDf, level2LabelsDf, level1LabelsDf = readSemEvalData()

    # convert training data to TER format
    if level == "0":
        level0TrainDf, level0TestDf = convertSemEvalDataToTerFormat(argumentsDf, level2LabelsDf, level="0")
        level1TrainDf, level1TestDf = convertSemEvalDataToTerFormat(argumentsDf, level1LabelsDf, level="1")
        level2TrainDf, level2TestDf = convertSemEvalDataToTerFormat(argumentsDf, level2LabelsDf, level="2")
        trainDf = pd.concat([level0TrainDf, level1TrainDf, level2TrainDf], ignore_index=True)
        testDf = pd.concat([level0TestDf, level1TestDf, level2TestDf], ignore_index=True)

    if level == "1":
        level1TrainDf, level1TestDf = convertSemEvalDataToTerFormat(argumentsDf, level1LabelsDf, level="1")
        level2TrainDf, level2TestDf = convertSemEvalDataToTerFormat(argumentsDf, level2LabelsDf, level="2")
        trainDf = pd.concat([level1TrainDf, level2TrainDf], ignore_index=True)
        testDf = pd.concat([level1TestDf, level2TestDf], ignore_index=True)

    if level == "2":
        trainDf, testDf = convertSemEvalDataToTerFormat(argumentsDf, level2LabelsDf, level="2")

    # get a subsample of the training data for pilot run
    if test == "yes":
        trainDf = trainDf.sample(100, random_state=1)
        evalDf = testDf.loc[testDf['Hypothesis'] == testDf['Hypothesis'].tolist()[0]]
    else:
        evalDf = testDf

    print(trainDf)
    print("next")
    print(evalDf)

    if batchSize is None:
        return

    # tokenize the datasets and prepare the data in Dataset class
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    trainEncodings, trainLabels = tokenize_data(trainDf, tokenizer)
    evalEncodings, evalLabels = tokenize_data(evalDf, tokenizer)
    trainDataset = semEvalTerDataset(trainEncodings, trainLabels)
    evalDataset = semEvalTerDataset(evalEncodings, evalLabels)

    # # train BERT
    if device == "hpc":
        outputDir = '/hpc/uu_cs_nlpsoc/data/qixiang/proj_semeval23_task4/terLevel' + level + 'Results'
    else:
        outputDir = './terLevel' + level + 'Results'

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    training_args = TrainingArguments(
        output_dir=outputDir,          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=batchSize,  # batch size per device during training
        per_device_eval_batch_size=128,   # batch size for evaluation
        warmup_steps=250,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        metric_for_best_model='loss',
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        save_strategy='epoch'
    )

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainDataset,
        eval_dataset=evalDataset
    )

    trainer.train()

if __name__ == '__main__':
    main()