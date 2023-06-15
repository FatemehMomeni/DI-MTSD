import torch
import torch.nn as nn
from transformers import AutoModel, DistilBertModel


# BERT/BERTweet
class stance_classifier(nn.Module):

  def __init__(self, num_labels):
    super(stance_classifier, self).__init__()
    
    self.dropout = nn.Dropout(0.)
    self.relu = nn.ReLU()
    self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    self.bert.pooler = None
    self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
    self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
    
  def forward(self, input_ids_tar, atten_masks_tar, input_ids_txt, atten_masks_txt, x_input_embeds_tar):
    
    if x_input_embeds_tar == None:
      output_tar = self.bert(input_ids=input_ids_tar, attention_mask=atten_masks_tar, output_hidden_states=True)
      output_txt = self.bert(input_ids=input_ids_txt, attention_mask=atten_masks_txt, output_hidden_states=True)
      out_tar = output_tar[1][0] # hiden states, initial embedding
      out_txt = output_txt[1][0]
      return out_tar, out_txt

    else:
      output = self.bert(inputs_embeds=x_input_embeds_tar)    
      cls = output[0][:,0] # last hidden state
      query = self.dropout(cls)
      linear = self.relu(self.linear(query))
      out = self.out(linear)        
      return out
