import preprocessor as p 
import re
import wordninja
import csv
import pandas as pd

# Data Loading
def load_data(filename,usecols,col,dataset_name):

    filename = [filename]
    concat_text = pd.DataFrame()
    raw_text = pd.read_csv(filename[0],usecols=[0], encoding='ISO-8859-1')
    raw_label = pd.read_csv(filename[0],usecols=[usecols[0]], encoding='ISO-8859-1')
    raw_target = pd.read_csv(filename[0],usecols=[usecols[1]], encoding='ISO-8859-1')
    related_target1 = pd.read_csv(filename[0],usecols=[3], encoding='ISO-8859-1')
    related_target2 = pd.read_csv(filename[0],usecols=[4], encoding='ISO-8859-1')
    related_target3 = pd.read_csv(filename[0],usecols=[5], encoding='ISO-8859-1')    
    if dataset_name in ['mt','semeval','am','covid','all']:
        label = pd.DataFrame.replace(raw_label,['FAVOR','NONE','AGAINST'], [2,1,0])
    elif dataset_name in ['wtwt']:
        label = pd.DataFrame.replace(raw_label,['support','comment','refute','unrelated'], [2,1,0,3])
    concat_text = pd.concat([raw_text, label, raw_target, related_target1, related_target2, related_target3], axis=1)
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
    x_related_target1 = concat_text['RelatedTarget1'].values.tolist()
    x_related_target2 = concat_text['RelatedTarget2'].values.tolist()
    x_related_target3 = concat_text['RelatedTarget3'].values.tolist()    
    clean_data = [None for _ in range(len(raw_data))]
    
    for i in range(len(raw_data)):
        clean_data[i] = data_clean(raw_data[i],norm_dict) # clean each tweet text [['word1','word2'],[...],...]
        x_target[i] = data_clean(x_target[i],norm_dict)      
        x_related_target1[i] = data_clean(x_related_target1[i],norm_dict)
        x_related_target2[i] = data_clean(x_related_target2[i],norm_dict)
        x_related_target3[i] = data_clean(x_related_target3[i],norm_dict)       
    
    return clean_data,label,x_target,x_related_target1,x_related_target2,x_related_target3
