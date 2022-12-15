from utils import  readSemEvalData
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 0)

argumentsDf, level2LabelsDf, level1LabelsDf = readSemEvalData()
print(argumentsDf)