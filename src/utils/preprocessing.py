import preprocessor as p 
import re
import wordninja
import csv
import pandas as pd

# import statistics
# import pandas as pd
# import numpy as np
# import string
# from sentence_transformers import SentenceTransformer, util
# from sklearn.feature_extraction.text import CountVectorizer

# def related_targets(clean_data, x_target):
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     targets = list()
#     related_terms = dict()
#     for tar in x_target:      
#       tar_phrase = ''      
#       for tar_word in tar:        
#         tar_phrase = tar_phrase + tar_word + ' '        
#       tar_phrase = tar_phrase.rstrip()      
#       if tar_phrase not in targets:
#         targets.append(tar_phrase)
#         related_terms[tar_phrase] = list()
    
#     punct = set(string.punctuation)
#     tokens = list()
#     for msg in clean_data:      
#       for word in msg:        
#         if word not in punct:
#           tokens.append(word)

#     msg_embeddings = model.encode(tokens)
#     for tar in targets:
#         target_embedding = model.encode(tar)
#         cos_sim = util.cos_sim(target_embedding, msg_embeddings)

#         similars = list()
#         for arr in cos_sim:
#             for i, score in enumerate(arr):
#                 if score >= 0.7:
#                     similars.append([tokens[i], score])
#         similars = [x for x in similars if similars.count(x) == 1]
#         sorted_similars = sorted(similars, key=lambda x: x[1], reverse=True)        

#         for token_score in sorted_similars:
#             related_terms[tar].append([token_score[0]])  

#     for key,value in related_terms.items():
#         if len(value) == 3:
#             related_terms[key] = value[:3]
#         else:
#             while len(value) < 3:
#                 related_terms[key].append([k for k in key.split()])
#         if type(key) == str:
#           new_key = key.split()
#           new_key = tuple(new_key)
#           related_terms[new_key] = related_terms.pop(key)
#         else:
#           break
        
#     return related_terms

# Data Loading
def load_data(filename,usecols,col,dataset_name):

    filename = [filename]
    concat_text = pd.DataFrame()
    raw_text = pd.read_csv(filename[0],usecols=[0], encoding='ISO-8859-1')
    raw_label = pd.read_csv(filename[0],usecols=[usecols[0]], encoding='ISO-8859-1')
    raw_target = pd.read_csv(filename[0],usecols=[usecols[1]], encoding='ISO-8859-1')
    if dataset_name in ['mt','semeval','am','covid','all']:
        label = pd.DataFrame.replace(raw_label,['FAVOR','NONE','AGAINST'], [2,1,0])
    elif dataset_name in ['wtwt']:
        label = pd.DataFrame.replace(raw_label,['support','comment','refute','unrelated'], [2,1,0,3])
    concat_text = pd.concat([raw_text, label, raw_target], axis=1)
    concat_text.rename(columns={'Stance 1':'Stance','Target 1':'Target','Stance 2':'Stance','Target 2':'Target'}, 
                       inplace=True)
    concat_text = concat_text[concat_text.Stance != 3] # remove 'unrelated' label of WT-WT
    
    return concat_text


# Data Cleaning
def data_clean(strings,norm_dict):
    
    p.set_options(p.OPT.URL,p.OPT.EMOJI,p.OPT.RESERVED)
    clean_data = p.clean(strings) # using lib to clean URL,hashtags...
    clean_data = re.sub(r"#SemST", "", clean_data)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+",clean_data)
    clean_data = [[x.lower()] for x in clean_data]
    
    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict.keys():
            clean_data[i] = norm_dict[clean_data[i][0]].split()
            continue
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0]) # separate hashtags
    clean_data = [j for i in clean_data for j in i]

    return clean_data


# Clean All Data
def clean_all(filename,col,dataset_name,norm_dict):
    
    usecols = [2,1] if col == "Stance1" else [4,3]
    concat_text = load_data(filename,usecols,col,dataset_name) # load all data as DataFrame type
    raw_data = concat_text['Tweet'].values.tolist() # convert DataFrame to list ['string','string',...]
    label = concat_text['Stance'].values.tolist()
    x_target = concat_text['Target'].values.tolist()
    clean_data = [None for _ in range(len(raw_data))]
    
    for i in range(len(raw_data)):
        clean_data[i] = data_clean(raw_data[i],norm_dict) # clean each tweet text [['word1','word2'],[...],...]
        x_target[i] = data_clean(x_target[i],norm_dict)
    
    return clean_data,label,x_target
