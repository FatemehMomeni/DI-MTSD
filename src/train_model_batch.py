import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import argparse
import json
import gc
import utils.preprocessing_batch as pp
import utils.data_helper_batch as dh
from transformers import AdamW
from utils import modeling, model_eval


def run_classifier():

  parser = argparse.ArgumentParser()
  parser.add_argument("--lr", type=float, default=2e-5)
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--dropout", type=float, default=0.)
  parser.add_argument("--alpha", type=float, default=0.7)
  args = parser.parse_args()

  random_seeds = [1]
  lr = args.lr
  batch_size = args.batch_size
  total_epoch = args.epochs
  dropout = args.dropout
  alpha = args.alpha

  target_num = 3  # 'all': 3 (generalization), 4 (srq)
  eval_batch = True
  
  # create normalization dictionary for preprocessing
  with open("/content/DI-MTSD/src/noslang_data.json", "r") as f:
    data1 = json.load(f)
  data2 = {}
  with open("/content/DI-MTSD/src/emnlp_dict.txt","r") as f:
    lines = f.readlines()
    for line in lines:
      row = line.split('\t')
      data2[row[0]] = row[1].rstrip()
  normalization_dict = {**data1,**data2}

  best_result, best_val = [], []
  for seed in random_seeds:    
    print("current random seed: ", seed)

    filename1 = '/content/DI-MTSD/Dataset/train_related.csv'
    filename2 = '/content/DI-MTSD/Dataset/val_related.csv'
    filename3 = '/content/DI-MTSD/Dataset/test_gen_related.csv'
    
    x_train,y_train,x_train_target = pp.clean_all(filename1,normalization_dict)
    x_val,y_val,x_val_target = pp.clean_all(filename2,normalization_dict)
    x_test,y_test,x_test_target = pp.clean_all(filename3,normalization_dict)
      
    x_train_all = [x_train,y_train,x_train_target]
    x_val_all = [x_val,y_val,x_val_target]
    x_test_all = [x_test,y_test,x_test_target]

    # set up the random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 

    num_labels = len(set(y_train))  # Favor, Against and None len(set(y_train))
    model = modeling.stance_classifier(num_labels).cuda()
    
    # prepare for model
    train_loader, train_loader_distil = dh.data_helper_bert(x_train_all, batch_size, 'train')
    val_loader, val_labels = dh.data_helper_bert(x_val_all, batch_size, 'val')
    test_loader, test_labels = dh.data_helper_bert(x_test_all, batch_size, 'test')

    for n, p in model.named_parameters():
      if "distilbert.embeddings" in n:
        p.requires_grad = False

    optimizer_grouped_parameters = [
      {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')] , 'lr': lr},
      {'params': [p for n, p in model.named_parameters() if n.startswith('bert.pooler')] , 'lr': 1e-3},
      {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': 1e-3},
      {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': 1e-3}
    ]

    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    sum_loss, sum_loss2 = [], []
    val_f1_average = []
    train_preds_distill, train_cls_distill = [], []
    test_f1_average = [[] for _ in range(target_num)]

    for epoch in range(total_epoch):
      print('Epoch:', epoch)
      train_loss, train_loss2 = [], []
      model.train()

      for input_ids_tar, atten_mask_tar, input_ids_txt, atten_mask_txt, y in train_loader:           
        tar_embeds, txt_embeds = model(input_ids_tar, atten_mask_tar, input_ids_txt, atten_mask_txt, None)        
        embeds = torch.cat((tar_embeds, txt_embeds), 1)                
        optimizer.zero_grad()
        output1 = model(None, None, None, None, embeds)
        loss = loss_function(output1, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        train_loss.append(loss.item())
      sum_loss.append(sum(train_loss)/len(x_train))
      print(sum_loss[epoch])

      # train evaluation
      model.eval()
      train_preds = []
      with torch.no_grad():
        for input_ids_tar, atten_mask_tar, input_ids_txt, atten_mask_txt, y in train_loader_distil:
          tar_embeds, txt_embeds = model(input_ids_tar, atten_mask_tar, input_ids_txt, atten_mask_txt, None)          
          embeds = torch.cat((tar_embeds, txt_embeds), 1)
          output1 = model(None, None, None, None, embeds)
          train_preds.append(output1)
        preds = torch.cat(train_preds, 0)
        train_preds_distill.append(preds)
        print("The size of train_preds is: ", preds.size())

      # evaluation on val set
      model.eval()
      val_preds = []
      with torch.no_grad():
        for input_ids_tar, atten_mask_tar, input_ids_txt, atten_mask_txt, y in val_loader:
          tar_embeds, txt_embeds = model(input_ids_tar, atten_mask_tar, input_ids_txt, atten_mask_txt, None)          
          embeds = torch.cat((tar_embeds, txt_embeds), 1)
          pred1 = model(None, None, None, None, embeds)
          val_preds.append(pred1)
        pred1 = torch.cat(val_preds, 0)
        acc, f1_average, precision, recall = model_eval.compute_f1(pred1, val_labels)
        val_f1_average.append(f1_average)

      # evaluation on test set
      y_test_list = dh.sep_test_set(test_labels)
      with torch.no_grad():
        test_preds = []
        for input_ids_tar, atten_mask_tar, input_ids_txt, atten_mask_txt, y in test_loader:
          tar_embeds, txt_embeds = model(input_ids_tar, atten_mask_tar, input_ids_txt, atten_mask_txt, None)          
          embeds = torch.cat((tar_embeds, txt_embeds), 1)
          pred1 = model(None, None, None, None, embeds)
          test_preds.append(pred1)
        pred1 = torch.cat(test_preds, 0)
        pred1_list = dh.sep_test_set(pred1)

        test_preds = []
        for ind in range(len(y_test_list)):
          pred1 = pred1_list[ind]
          test_preds.append(pred1)
          acc, f1_average, precision, recall = model_eval.compute_f1(pred1, y_test_list[ind])
          test_f1_average[ind].append(f1_average)

    # model that performs best on the dev set is evaluated on the test set
    best_epoch = [index for index,v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
    best_result.append([f1[best_epoch] for f1 in test_f1_average])

    print("******************************************")
    print("dev results with seed {} on all epochs".format(seed))
    print(val_f1_average)
    best_val.append(val_f1_average[best_epoch])
    print("******************************************")
    print("test results with seed {} on all epochs".format(seed))
    print(test_f1_average)
    print("******************************************")
    print(max(best_result))
    print(best_result)


if __name__ == "__main__":
  run_classifier()
