import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer
    

# Tokenization
def convert_data_to_ids(tokenizer, target, related_target1, related_target2, related_target3, text, y, mode):
    
    input_ids, seg_ids, attention_masks, sent_len, labels = [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)]
    if mode != 'test':
      target = target[:10048]
      related_target1 = related_target1[10048:20032]
      related_target2 = related_target2[20032:30016]
      related_target3 = related_target3[30016:]
      indices = [0, 10048, 20032, 30016, 39996]
    else :
      target = target[:3616]
      related_target1 = related_target1[3616:7136]
      related_target2 = related_target2[7136:10656]
      related_target3 = related_target3[10656:]
      indices = [0, 3616, 7136, 10656, 14167]
    targets = [target, related_target1, related_target2, related_target3]

    for i in range(4):
      for tar, sent in zip(targets[i], text[indices[i]:indices[i+1]]):
          encoded_dict = tokenizer.encode_plus(
                              ' '.join(tar),
                              ' '.join(sent),             # Sentence to encode.
                              add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                              max_length = 128,           # Pad & truncate all sentences.
                              padding = 'max_length',
                              return_attention_mask = True,   # Construct attn. masks.
                              truncation = True,
                        )
      
          # Add the encoded sentence to the list.    
          input_ids[i].append(encoded_dict['input_ids'])  
          seg_ids[i].append(encoded_dict['token_type_ids'])
          attention_masks[i].append(encoded_dict['attention_mask'])
          sent_len[i].append(sum(encoded_dict['attention_mask']))      
      
    for i in range(4):
      labels.append(y[indices[i]:indices[i+1]])

    return input_ids, seg_ids, attention_masks, sent_len, labels


# BERT/BERTweet tokenizer    
def data_helper_bert(x_train_all,x_val_all,x_test_all,main_task_name,model_select, mode):
    
    print('Loading data')   
    x_train,y_train,x_train_target,x_train_related_target1,x_train_related_target2,x_train_related_target3 = x_train_all[0],x_train_all[1],x_train_all[2],x_train_all[3],x_train_all[4],x_train_all[5]
    x_val,y_val,x_val_target,x_val_related_target1,x_val_related_target2,x_val_related_target3 = x_val_all[0],x_val_all[1],x_val_all[2],x_val_all[3],x_val_all[4],x_val_all[5]
    x_test,y_test,x_test_target,x_test_related_target1,x_test_related_target2,x_test_related_target3 = x_test_all[0],x_test_all[1],x_test_all[2],x_test_all[3],x_test_all[4],x_test_all[5]
    print("Length of original x_train: %d"%(len(x_train)))
    print("Length of original x_val: %d, the sum is: %d"%(len(x_val), sum(y_val)))
    print("Length of original x_test: %d, the sum is: %d"%(len(x_test), sum(y_test)))
    
    # get the tokenizer
    if model_select == 'Bertweet':
        tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    elif model_select == 'Bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # tokenization
    x_train_input_ids, x_train_seg_ids, x_train_atten_masks, x_train_len, train_labels = \
                    convert_data_to_ids(tokenizer, x_train_target, x_train_related_target1,x_train_related_target2,x_train_related_target3, x_train, y_train, 'train')
    x_val_input_ids, x_val_seg_ids, x_val_atten_masks, x_val_len, val_labels = \
                    convert_data_to_ids(tokenizer, x_val_target, x_val_related_target1,x_val_related_target2,x_val_related_target3, x_val, y_val, 'val')
    x_test_input_ids, x_test_seg_ids, x_test_atten_masks, x_test_len, test_labels = \
                    convert_data_to_ids(tokenizer, x_test_target, x_test_related_target1,x_test_related_target2,x_test_related_target3, x_test, y_test, 'test')
    
    x_train_all = [x_train_input_ids,x_train_seg_ids,x_train_atten_masks,train_labels,x_train_len]
    x_val_all = [x_val_input_ids,x_val_seg_ids,x_val_atten_masks,val_labels,x_val_len]
    x_test_all = [x_test_input_ids,x_test_seg_ids,x_test_atten_masks,test_labels,x_test_len]
    
    print(len(x_train), sum(y_train))
    print("Length of final x_train: %d"%(len(x_train)))
    
    return x_train_all,x_val_all,x_test_all


def data_loader(x_all, batch_size, model_select, mode, model_name, **kwargs):
    
    x_input_ids, x_seg_ids, x_atten_masks, y , x_len = [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)]
    for i in range(4):
      x_input_ids.append(torch.tensor(x_all[0][i], dtype=torch.long).cuda())
      x_seg_ids.append(torch.tensor(x_all[1][i], dtype=torch.long).cuda())
      x_atten_masks.appendtorch.tensor(x_all[2][i], dtype=torch.long).cuda())
      y.append(torch.tensor(x_all[3][i], dtype=torch.long).cuda())
      x_len.append(torch.tensor(x_all[4][i], dtype=torch.long).cuda())

    if model_name == 'student' and mode == 'train':
        y2 = torch.tensor(kwargs['y_train2'], dtype=torch.float).cuda()  # load teacher predictions
        y2_labels = list()
        indices = [0, 10048, 20032, 30016, 39996]
        for i in range(4):
          y2_labels.append(y2[indices[i]:indices[i+1]])        
        tensor_loader = TensorDataset(x_input_ids,x_seg_ids,x_atten_masks,y,x_len,y2_labels)
    else:
        tensor_loader = TensorDataset(x_input_ids,x_seg_ids,x_atten_masks,y,x_len)

    if mode == 'train':
        data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
        data_loader_distill = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)

        return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader, data_loader_distill
    else:
        data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)

        return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader


def sep_test_set(input_data,dataset_name):
    
    # split the combined test set for each target    
    if dataset_name == 'all':
        # generalization test set
        data_list = [input_data[:10238], input_data[10238:12204], input_data[12204:]]
        # diffetent test set
        #data_list = [input_data[:2020], input_data[2020:3266], input_data[3266:4484], input_data[4484:]]
    
    return data_list
