from utils import readSemEvalData, convertSemEvalDataToTerFormat
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 0)

# read training data
argumentsDf, _, _, testDf = readSemEvalData(long_format=True)
print(argumentsDf)
print(testDf)

#
# level2TestDf = convertSemEvalDataToTerFormat(testDf, value_level="2", test=True)
# print(level2TestDf)