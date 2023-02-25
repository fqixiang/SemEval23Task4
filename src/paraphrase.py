from utils import read_train_and_val_data, t5_vamsi_paraphraser, convert_data_to_nli_format
import pandas as pd

values = ['Stimulation', 'Hedonism', 'Power: dominance', 'Face', 'Conformity: interpersonal', 'Humility']
arguments_train_df, _, level2_labels_train_df, _ = read_train_and_val_data(long_format=False)
index = (level2_labels_train_df[values].sum(axis=1)) > 0
argument_ids = arguments_train_df.loc[index]['Argument ID'].tolist()

t5_vamsi_paraphraser(arguments_train_df, argument_ids)

# print(pd.read_csv('../../data/raw/paraphrases.tsv', sep='\t'))