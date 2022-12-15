from sklearn.metrics import classification_report
import pandas as pd

report = classification_report([0,1,1,2,3,3,1,2], [0,0,1,2,3,3,1,2], output_dict = True)
report = pd.DataFrame(report).transpose()
# print(report)

classificationResults = {}
report = classification_report([1,1,1], [0,1,1], output_dict = True)
classificationResults['value0'] = report['1']
print(classificationResults)
#
# # report = pd.DataFrame(report['1']).transpose()
classificationDf = pd.DataFrame.from_dict(classificationResults, orient='index')
print(classificationDf)
classificationDf.append(classificationDf, ignore_index=True)