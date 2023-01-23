from utils import readSemEvalData, convertSemEvalDataToTerFormat, semEvalTerDataset, tokenize_data
import pandas as pd
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import classification_report
import argparse
import torch
import time
import os
torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level",
                        type=str,
                        default=None)
    parser.add_argument("--model_path",
                        type=str,
                        default=None)
    parser.add_argument("--batch_size",
                        type=int,
                        default=512)

    args = parser.parse_args()

    level = args.level
    modelPath = args.model_path

    # read test data
    argumentsDf, labelsDf, _, _ = readSemEvalData()
    _, level2TestDf = convertSemEvalDataToTerFormat(argumentsDf, labelsDf, value_level="2")
    print(level2TestDf)

    # load trained BERT
    outputDir = '../../results/output/prediction/terTrainedWithLevel' + level + 'Values'
    trainingArgs = TrainingArguments(
        output_dir=outputDir,          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=512,   # batch size for evaluation
        warmup_steps=250,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        metric_for_best_model='loss',
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        save_strategy='epoch'
    )

    model = AutoModelForSequenceClassification.from_pretrained(modelPath, num_labels=2)
    trainer = Trainer(model=model,
                      args=trainingArgs)

    # tokenize the datasets and prepare the data in Dataset class
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    testEncodings, testLabels = tokenize_data(level2TestDf, tokenizer, test=True)
    testDataset = semEvalTerDataset(testEncodings, testLabels)
    preds = np.argmax(trainer.predict(testDataset)[0], axis=1)
    level2TestDf['Entailment'] = preds

    predictionDf = pd.pivot(level2TestDf, index="Argument ID", columns="Hypothesis", values="Entailment")

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    timestamp = time.strftime("%Y%m%d%H%M")
    resultsName = outputDir + "/terEvaluationWithLevel" + level + "Hypotheses" + timestamp + ".tsv"
    predictionDf.to_csv(resultsName, sep='\t')

if __name__ == '__main__':
    main()