#Import required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
from transformers import AutoModelForSequenceClassification
import os
torch.cuda.empty_cache()

#DEFINE INPUT AND OUTPUT PATHS

arguments_path = '../data/touche23/arguments-training.tsv'
labels_path = '../data/touche23/labels-training.tsv'
para_path = '/hpc/uu_cs_nlpsoc/data/qixiang/proj_semeval23_task4/paraphrase/'
testresult = para_path + 'test_result.csv'

if not os.path.exists(para_path):
    os.mkdir(para_path)

#import arguments and labels
arguments = pd.read_csv(arguments_path, sep = '\t')
labels = pd.read_csv(labels_path, sep = '\t')

#Store category labels 
categories = list(labels.columns.values[1:])

#Setting up the paraphraser
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name,
                                             cache_dir=para_path)
model = PegasusForConditionalGeneration.from_pretrained(model_name, cache_dir=para_path).to(torch_device)

#setting up the model
def get_response(input_text,num_return_sequences):
    try:
        batch = tokenizer(input_text,truncation=True,padding='longest', return_tensors="pt").to(torch_device)
    except:
        print(input_text)
        exit()

    translated = model.generate(**batch, max_length=180, num_beams=10, num_return_sequences=num_return_sequences,
                                temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

#Define function for data preprocessing

def concat(df):

  """This function concatenates the statements and
  removes punctuation and special characters
  """
  #Concatenate conclusion, stance, and premise
  #df['arg'] = #df['Conclusion'] + ' ' + df['Stance'] + ' ' + df['Premise']

  df['arg'] = df['Premise']
  #remove punctuation and special characters
  df['arg'] = df['arg'].str.replace('[^\w\s]', '', regex=True)

  #convert to lower case
  df['arg'] = df['arg'].str.lower()

  return df

def augment(labels, train):
  """This function augments the training data set with paraphrases

  """
  #Paraphrase Hedonism
  # para = labels[labels['Hedonism'] == 1]["Argument ID"]
  para = labels[(labels['Hedonism'] == 1) |
               (labels['Face'] == 1) |
               (labels['Power: dominance'] == 1) |
               (labels['Conformity: interpersonal'] == 1) |
               (labels['Humility'] == 1) |
               (labels['Benevolence: dependability'] == 1)
               ]["Argument ID"]
  para = pd.merge(train, para, how = 'right', 
                         left_on = 'Argument ID', right_on = 'Argument ID')

  #Define function for paraphrasing 
  def paraphrase(para):
    paraphrase = []
    id = []
    for i in range(len(para)):
      r = get_response(str(para.loc[i, "arg"]), 5)
      paraphrase.append(r)
      paraphrase.append(para.loc[i, "arg"])
      counter = 0
      for x in range(len(r)):
        counter += 1
        #if counter == 1: 
        #  id.append(str(para.loc[i, "Argument ID"]))
        #else: 
        id.append(str(para.loc[i, "Argument ID"]) + '_' + str(counter))
    return paraphrase, id
    
  parap, id = paraphrase(para)
  
  def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

  flat_list = flatten_list(parap)
  
  frame = pd.DataFrame([flat_list, id]).T
  frame = frame.rename(columns = {1: "Argument ID", 0: "arg"})
  #argid = frame["Argument ID"]

  frame["Stem"] = frame["Argument ID"].str[:6]

  args = pd.merge(frame, labels, how = 'left', left_on = 'Stem', right_on = 'Argument ID')
  args = args.drop(["Stem", "Argument ID_y"], axis = 1)
  args = args.rename(columns = {'Argument ID_x' : 'Argument ID'})
  
  
  #Merge with labels
  full = pd.merge(train, labels, on = 'Argument ID')
  full = full.drop(columns = ['Conclusion', 'Stance', 'Premise'])
  joined = pd.concat([full, args], join = 'outer')
  joined = joined.dropna(axis = 0)
 
 
  return joined

def validation_set(train):
  x_train = train["arg"]
  y_train = train.drop(columns = ['Argument ID', 'arg'])
  x_tr,x_val,y_tr,y_val = train_test_split(x_train, y_train, test_size = 0.1,
                                             random_state = 42, 
                                             shuffle = True)
  return x_tr, x_val, y_tr, y_val

#Defining main function
def main(arguments, labels):
  """Main function, preprocesses the data, paraphrases, and outputs 
  training, validation, and test data set
  """
  
  df = concat(arguments)
  # df = df.sample(frac = 0.01, replace = False, random_state = 42)
  train, test = train_test_split(df, random_state = 42, test_size = 0.10, shuffle = True)
  train = augment(labels, train)
  todrop = train[train["arg"].str.contains("\\bnan\\b", regex=True)]['Argument ID']
  todrop.sort_index(inplace = True)
  todrop = pd.DataFrame(todrop)
  train = train[-train['Argument ID'].isin(todrop['Argument ID'])]
  x_tr, x_val, y_tr, y_val = validation_set(train)
  test2 = pd.merge(test, labels, left_on = 'Argument ID', right_on = 'Argument ID', how = 'left')
  #Apply vectorizer to test data
  x_test = test2["arg"]
  #Construct y_test
  y_test = test2.drop(columns = ['Argument ID', 'arg', 'Conclusion', 'Stance', 'Premise'])

  return x_tr, x_val, y_tr, y_val, x_test, y_test

x_tr, x_val, y_tr, y_val, x_test, y_test = main(arguments, labels)

#Save result of paraphrasing
x_tr.to_csv(para_path + 'x_tr.csv')
x_val.to_csv(para_path +'x_val.csv')
y_tr.to_csv(para_path+ 'y_tr.csv')
y_val.to_csv(para_path +'y_val.csv')
x_test.to_csv(para_path +'x_test.csv')
y_test.to_csv(para_path +'y_test.csv')

#
#x_tr = pd.read_csv('x_tr.csv')
#x_val = pd.read_csv('x_val.csv')
#y_tr = pd.read_csv('y_tr.csv')
#y_val = pd.read_csv('y_val.csv')
#x_test = pd.read_csv('x_test.csv')
#y_test = pd.read_csv('y_test.csv')

#Create untokenized training, validating, and testing data set
#train = pd.merge(x_tr, y_tr, left_on = x_tr['Unnamed: 0'], right_on = y_tr['Unnamed: 0']).drop(columns = ['Unnamed: 0_y', 'Unnamed: 0_x'])
#val  = pd.merge(x_val, y_val, left_on = x_val['Unnamed: 0'], right_on = y_val['Unnamed: 0']).drop(columns = ['Unnamed: 0_y', 'Unnamed: 0_x'])
#test = pd.merge(x_test, y_test, left_on=x_test['Unnamed: 0'], right_on = y_test['Unnamed: 0']).drop(columns = ['Unnamed: 0_y', 'Unnamed: 0_x'])

#Create untokenized training, validating, and testing data set
train = pd.merge(x_tr, y_tr, left_on = x_tr.index, right_on = y_tr.index)
val  = pd.merge(x_val, y_val, left_on = x_val.index, right_on = y_val.index)
test = pd.merge(x_test, y_test, left_on=x_test.index, right_on = y_test.index)

#Convert datasets to a DatasetDict() for torch
ds = DatasetDict()
train = Dataset.from_pandas(train)
val = Dataset.from_pandas(val)
test = Dataset.from_pandas(test)
ds['train'] = train
ds['validation'] = val
ds['test'] = test

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
                                          cache_dir=para_path)

def preprocess_data(examples):
  # take a batch of texts
  text = examples["arg"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=180)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in categories}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(categories)))
  # fill numpy array
  for idx, label in enumerate(categories):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding

encoded_dataset = ds.map(preprocess_data, batched=True, remove_columns=ds['train'].column_names)
encoded_dataset.set_format("torch")

id2label = {idx:label for idx, label in enumerate(categories)}
label2id = {label:idx for idx, label in enumerate(categories)}

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                           cache_dir=para_path,
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(categories),
                                                           id2label=id2label,
                                                           label2id=label2id)

batch_size = 128
metric_name = "f1"
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir=para_path,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=15,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'macro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_macro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics 
)

trainer.train()

def testing():
  """This function evaluates the performance of the BERT model on the
  unseen test data. It outputs as csv that contains precision, recall, 
  and F1 scores for all value categories and overall.
  """
  
  preds = trainer.predict(encoded_dataset['test'])
  def logit2prob(logit):
    prob = np.exp(logit) / (1 + np.exp(logit))
    return prob
  prob = []
  length = len(preds.predictions)
  for row in range(length):
    l = []
    for i in preds.predictions[row]:
      p = logit2prob(i)
      if p < 0.5:
        l.append(0)
      else: l.append(1)
    prob.append(l)
  y_preds = pd.DataFrame(prob)
  y_preds.columns = categories
  classifications = classification_report(y_test, y_preds, output_dict = True)
  classifications_frame = pd.DataFrame(classifications).T
  classifications_frame.to_csv(testresult)

testing()
