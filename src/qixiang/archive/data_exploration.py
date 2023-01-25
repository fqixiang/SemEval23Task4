import pandas as pd
from utils import readSemEvalData, convertSemEvalDataToTerFormat
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 0)

# read training data
# argumentsDf, level2LabelsDf, level1LabelsDf = readSemEvalData()
# # print(level2LabelsDf)
# print(level1LabelsDf)
# print(level2LabelsDf)
#
# # convert training data to TER format
# trainDf, testDf = convertSemEvalDataToTerFormat(argumentsDf, level2LabelsDf, level="0")
# trainDf, testDf = convertSemEvalDataToTerFormat(argumentsDf, level1LabelsDf, level="1")
# trainDf, testDf = convertSemEvalDataToTerFormat(argumentsDf, level2LabelsDf, level="2")

# print(testDf)
#
# df = pd.DataFrame({"a": ["there is a banana .", "this is nan", "person named nan?", "proper"]})
# print(df["a"].str.contains("\\bnan\\b", regex=True))

# level 2 labels
def showValuePercentages(data_path):
    level2LabelsDf = pd.read_csv(data_path, sep='\t')
    print("Size:",level2LabelsDf.shape)
    level2Values = list(level2LabelsDf.columns)[1:]
    level2LabelsDf = pd.melt(level2LabelsDf, id_vars="Argument ID", value_vars=level2Values,
                             var_name='Value', value_name="Entailment")
    print(level2LabelsDf.groupby(["Value"]).mean())

showValuePercentages(data_path='../data/touche23/new/labels-training.tsv')
showValuePercentages(data_path='../data/touche23/new/labels-validation.tsv')
showValuePercentages(data_path='../data/touche23/new/labels-validation-zhihu.tsv')