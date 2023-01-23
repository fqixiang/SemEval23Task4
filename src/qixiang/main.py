from utils import read_train_and_val_data, convert_data_to_nli_format, tokenize_data, MyDataset, read_test_data
from transformers import AutoTokenizer


arguments_test, labels_test = read_test_data()
test = convert_data_to_nli_format(arguments_test, labels_test, definition="description")
print(test)

#
# arguments_train_df, arguments_val_df, level2_labels_train_df, level2_labels_val_df = read_semeval_data()
# train = convert_data_to_nli_format(arguments_train_df, level2_labels_train_df, definition="description")
# val = convert_data_to_nli_format(arguments_val_df, level2_labels_val_df, definition="description")
#
# #use samples
# train = train.sample(n=1000, random_state=1)
# val = val.sample(n=1000, random_state=1)
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# train_encodings, train_labels, max_length = tokenize_data(train, tokenizer, max_length=None)
# val_encodings, val_labels = tokenize_data(val, tokenizer, max_length=max_length)
# # print(max_length)
# # print(train_tokenized["input_ids"][0])
# # print(val_tokenized["input_ids"][0])
#
# train_dataset = MyDataset(train_encodings, train_labels)
# val_dataset = MyDataset(val_encodings, val_labels)