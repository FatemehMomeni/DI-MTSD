from numpy import indices
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer
    

# Tokenization
def convert_data_to_ids(tokenizer, target, related_target1, related_target2, related_target3, text, mode):
    
    input_ids, seg_ids, attention_masks, sent_len = [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)]
    if mode == 'test':
      target = target[:3520]
      related_target1 = related_target1[3520:7040]
      related_target2 = related_target2[7040:10560]
      related_target3 = related_target3[10560:14080]  
      indices = [0,3520,7040,10560,14080]    
    elif mode == 'train':
      target = target[:9984]
      related_target1 = related_target1[9984:19968]
      related_target2 = related_target2[19968:29952]
      related_target3 = related_target3[29952:39936]
      indices = [0,9984,19968,29952,39936]
    else:
      target = target[:1888]
      related_target1 = related_target1[1888:3776]
      related_target2 = related_target2[3776:5664]
      related_target3 = related_target3[5664:7552]
      indices = [0,1888,3776,5664,7552]
    targets = [target,related_target1,related_target2,related_target3]

    for z in range(len(targets)):  
      input_ids_tmp = list()
      seg_ids_tmp = list()
      attention_masks_tmp = list()
      sent_len_tmp = list()
      for tar, sent in zip(targets[z], text[indices[z]:indices[z+1]]):
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
          input_ids_tmp.append(encoded_dict['input_ids'])
          seg_ids_tmp.append(encoded_dict['token_type_ids'])
          attention_masks_tmp.append(encoded_dict['attention_mask'])
          sent_len_tmp.append(sum(encoded_dict['attention_mask']))
      input_ids[z] = input_ids_tmp
      seg_ids[z] = seg_ids_tmp
      attention_masks[z] = attention_masks_tmp
      sent_len[z] = sent_len_tmp
          
    return input_ids, seg_ids, attention_masks, sent_len


# BERT/BERTweet tokenizer    
def data_helper_bert(x_train_all,x_val_all,x_test_all,main_task_name,model_select):
    
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
    x_train_input_ids, x_train_seg_ids, x_train_atten_masks, x_train_len = \
                    convert_data_to_ids(tokenizer, x_train_target, x_train_related_target1,x_train_related_target2,x_train_related_target3, x_train, 'train')
    x_val_input_ids, x_val_seg_ids, x_val_atten_masks, x_val_len = \
                    convert_data_to_ids(tokenizer, x_val_target, x_val_related_target1,x_val_related_target2,x_val_related_target3, x_val, 'val')
    x_test_input_ids, x_test_seg_ids, x_test_atten_masks, x_test_len = \
                    convert_data_to_ids(tokenizer, x_test_target, x_test_related_target1,x_test_related_target2,x_test_related_target3, x_test, 'test')
      
    x_train_all = [x_train_input_ids,x_train_seg_ids,x_train_atten_masks,y_train,x_train_len]
    x_val_all = [x_val_input_ids,x_val_seg_ids,x_val_atten_masks,y_val,x_val_len]
    x_test_all = [x_test_input_ids,x_test_seg_ids,x_test_atten_masks,y_test,x_test_len]
    
    print(len(x_train), sum(y_train))
    print("Length of final x_train: %d"%(len(x_train)))
    
    return x_train_all,x_val_all,x_test_all


def data_loader(x_all, batch_size, model_select, mode, model_name, **kwargs):
    
    x_input_ids, x_seg_ids, x_atten_masks, x_len = [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)]
    labels,tensor_loader,data_loader,data_loader_distill = [[] for _ in range(4)],[[] for _ in range(4)],[[] for _ in range(4)],[[] for _ in range(4)]
    indices_train = [0,9984,19968,29952,39936]
    indices_test = [0,3520,7040,10560,14080]
    indices_val = [0,1888,3776,5664,7552] 

    for z in range(4):
      x_input_ids[z] = torch.tensor(x_all[0][z], dtype=torch.long).cuda()
      x_seg_ids[z] = torch.tensor(x_all[1][z], dtype=torch.long).cuda()
      x_atten_masks[z] = torch.tensor(x_all[2][z], dtype=torch.long).cuda()    
      x_len[z] = torch.tensor(x_all[4][z], dtype=torch.long).cuda()
    y = torch.tensor(x_all[3], dtype=torch.long).cuda()
    
    if model_name == 'student' and mode == 'train':
        y2 = torch.tensor(kwargs['y_train2'], dtype=torch.float).cuda()  # load teacher predictions
        for z in range(4):
          tensor_loader[z] = TensorDataset(x_input_ids[z],x_seg_ids[z],x_atten_masks[z],y[indices_train[z]:indices_train[z+1]],x_len[z],y2[indices_train[z]:indices_train[z+1]])
    else:
      if mode == 'train':
        ind = indices_train
      elif mode == 'test':
        ind = indices_test
      else:
        ind = indices_val
      for z in range(4):    
        tensor_loader[z] = TensorDataset(x_input_ids[z],x_seg_ids[z],x_atten_masks[z],y[ind[z]:ind[z+1]],x_len[z])        

    if mode == 'train':
      for z in range(4):
          data_loader[z] = DataLoader(tensor_loader[z], shuffle=True, batch_size=batch_size)
          data_loader_distill[z] = DataLoader(tensor_loader[z], shuffle=False, batch_size=batch_size)

      return x_input_ids, x_seg_ids, x_atten_masks, y[:indices_train[-1]], x_len, data_loader, data_loader_distill
    
    else:      
      for z in range(4):
        data_loader[z] = DataLoader(tensor_loader[z], shuffle=False, batch_size=batch_size)
      if mode == 'test':
        last = indices_test[-1]
      else:
        last = indices_val[-1]
      return x_input_ids, x_seg_ids, x_atten_masks, y[:last], x_len, data_loader


def sep_test_set(input_data,dataset_name):
        
    if dataset_name == 'all':
        # generalization test set
        data_list = [input_data[:10238], input_data[10238:12204], input_data[12204:]]
        # diffetent test set
        #data_list = [input_data[:2020], input_data[2020:3266], input_data[3266:4484], input_data[4484:]]
    
    return data_list
