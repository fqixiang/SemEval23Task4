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
    parser.add_argument("--modelPath",
                        type=str,
                        default=None)

    args = parser.parse_args()

    level = args.level
    modelPath = args.modelPath

    # read training data
    argumentsDf, labelsDf, _ = readSemEvalData()

    # convert level 2 test data to TER format
    _, level2TestDf = convertSemEvalDataToTerFormat(argumentsDf, labelsDf, level="2")
    level2TestDf = level2TestDf.sample(500, random_state=1)

    testDf = level2TestDf.rename(columns={'Hypothesis': 'Level2 Hypothesis'})

    # enrich level 2 test data with level 0 and 1 hypotheses
    dictionaryDfLevel1 = pd.read_csv('../data/touche23/semEvalDictionary - level1.tsv', sep='\t')
    dictionaryDfLevel0 = pd.read_csv('../data/touche23/semEvalDictionary - level0.tsv', sep='\t')

    testDf = pd.merge(testDf, dictionaryDfLevel1, how="left", on="Level2 Hypothesis")
    testDf = pd.merge(testDf, dictionaryDfLevel0, how="left", on=["Level1 ID", "Level1 Hypothesis"])

    # tokenize the datasets and prepare the data in Dataset class
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # load trained BERT
    outputDir = './terLevel' + level + 'Results'
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

    #evaluation from level0
    def evaluationReport(level):
        classificationResults = {}
        predictionResults = []
        for value in testDf["Level2 Hypothesis"].unique():
            print(value)
            testDfTemp = testDf.loc[testDf["Level2 Hypothesis"] == value]
            hypothesis = "Level" + level + " Hypothesis"
            testDfTemp = testDfTemp.rename(columns={hypothesis: 'Hypothesis'})
            testDfTemp = testDfTemp.drop_duplicates(subset=['Argument ID', 'Hypothesis'])
            testDfTemp = testDfTemp.reset_index(drop=True)

            testEncodingsTemp, testLabelsTemp = tokenize_data(testDfTemp, tokenizer)
            testDatasetTemp = semEvalTerDataset(testEncodingsTemp, testLabelsTemp)

            preds = np.argmax(trainer.predict(testDatasetTemp)[0], axis=1)
            testDfTemp['Prediction'] = preds

            preds = testDfTemp.groupby('Argument ID')['Prediction'].sum() > 0
            predsDf = preds.to_frame()
            predsDf = predsDf.reset_index()

            predsDf['Prediction'] = predsDf['Prediction'].astype(int)
            predictionResults.append(predsDf)

            predsDf = pd.merge(predsDf, level2TestDf.loc[level2TestDf['Hypothesis'] == value], how="left", on="Argument ID")

            report = classification_report(predsDf['Entailment'].tolist(), predsDf['Prediction'].tolist(),
                                           output_dict=True)

            classificationResults[value] = report['1']

        classificationDf = pd.DataFrame.from_dict(classificationResults, orient='index')
        classificationDf["Level"] = level
        classificationDf = classificationDf.reset_index()

        return classificationDf, predictionResults

    level0ClassificationDf,  level0PredictionResults = evaluationReport(level="0")
    level1ClassificationDf,  level1PredictionResults = evaluationReport(level="1")
    level2ClassificationDf,  level2PredictionResults = evaluationReport(level="2")

    classificationDf = pd.concat([level0ClassificationDf, level1ClassificationDf, level2ClassificationDf],
                                 ignore_index=True)

    print(level0PredictionResults)

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    timestamp = time.strftime("%Y%m%d%H%M")
    resultsName = outputDir + "/terResultsWithLevel" + level + "Hypotheses" + timestamp + ".tsv"
    classificationDf.to_csv(resultsName, sep='\t')

if __name__ == '__main__':
    main()