import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer


# BERT/BERTweet tokenizer    
def data_helper_bert(data, main_task_name, model_select):
    print('Loading data')
    if model_select == 'Bertweet':
        tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
        # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", normalization=True)
    elif model_select == 'Bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    input_ids, seg_ids, attention_masks, sent_len = [], [], [], []
    for i in range(len(data[0])):
        prompt = (f"The stance of text '{data[1][i]}' with respect to targets '{data[0][i].lower()}' or '{data[3][i]}' "
                  f"or '{data[4][i]}' or '{data[5][i]}' on domain '{data[6][i]}' is [MASK].")
        encoded_dict = tokenizer.encode_plus(prompt, add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                             max_length=128, padding='max_length',
                                             return_attention_mask=True,  # Construct attn. masks.
                                             truncation=True, )
    # for tar, sent in zip(data[0], data[1]):
    #     encoded_dict = tokenizer.encode_plus(tar, sent, add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    #                                          max_length=128,  # Pad & truncate all sentences.
    #                                          padding='max_length',
    #                                          return_attention_mask=True,  # Construct attn. masks.
    #                                          truncation=True, )
        input_ids.append(encoded_dict['input_ids'])
        seg_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        sent_len.append(sum(encoded_dict['attention_mask']))

    x_all = [input_ids, seg_ids, attention_masks, data[2], sent_len]
    # x_all = [input_ids, attention_masks, data[2], sent_len]
    return x_all


def data_loader(x_all, batch_size, model_select, mode, model_name, **kwargs):
    x_input_ids = torch.tensor(x_all[0], dtype=torch.long).cuda()
    x_seg_ids = torch.tensor(x_all[1], dtype=torch.long).cuda()
    x_atten_masks = torch.tensor(x_all[2], dtype=torch.long).cuda()
    y = torch.tensor(x_all[3], dtype=torch.long).cuda()
    x_len = torch.tensor(x_all[4], dtype=torch.long).cuda()

    if model_name == 'student' and mode == 'train':
        y2 = torch.tensor(kwargs['y_train2'], dtype=torch.float).cuda()  # load teacher predictions
        tensor_loader = TensorDataset(x_input_ids, x_seg_ids, x_atten_masks, y, x_len, y2)
        # tensor_loader = TensorDataset(x_input_ids, x_atten_masks, y, x_len, y2)
    else:
        tensor_loader = TensorDataset(x_input_ids, x_seg_ids, x_atten_masks, y, x_len)
        # tensor_loader = TensorDataset(x_input_ids, x_atten_masks, y, x_len)

    if mode == 'train':
        data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
        data_loader_distill = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
        return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader, data_loader_distill
        # return x_input_ids, x_atten_masks, y, x_len, data_loader, data_loader_distill
    else:
        data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
        return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader
        # return x_input_ids, x_atten_masks, y, x_len, data_loader


def sep_test_set(input_data, dataset_name):
    data_list = [input_data[:10238], input_data[10238:12204], input_data[12204:]]
    return data_list
