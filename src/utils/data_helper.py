import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertTokenizer


def data_helper_bert(data, batch_size, mode):
  print('Loading data')
  tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')    
  
  encoded_tar = tokenizer(data[0], max_length=128, padding='max_length', truncation=True)    
  encoded_txt = tokenizer(data[1], max_length=128, padding='max_length', truncation=True)
  
  input_ids_tar = torch.tensor(encoded_tar['input_ids']).cuda()
  atten_masks_tar = torch.tensor(encoded_tar['attention_mask']).cuda()
  input_ids_txt = torch.tensor(encoded_txt['input_ids']).cuda()
  atten_masks_txt = torch.tensor(encoded_txt['attention_mask']).cuda()
  y = torch.tensor(data[2]).cuda()
  
  tensor_loader = TensorDataset(input_ids_tar, atten_masks_tar, input_ids_txt, atten_masks_txt, y)
  data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
  
  if mode == 'train':
    data_loader_distil = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)    
    return data_loader, data_loader_distil
  else:
    return data_loader, y  


def sep_test_set(input_data):
  # generalization test set
  data_list = [input_data[:10238], input_data[10238:12204], input_data[12204:]]
  # diffetent test set
  #data_list = [input_data[:2020], input_data[2020:3266], input_data[3266:4484], input_data[4484:]]

  return data_list
