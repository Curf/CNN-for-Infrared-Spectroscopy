import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re


# In[275]:
r = './data/Shootout_2018_datasets.zip'
directory_to_extract_to = './data'

import zipfile
with zipfile.ZipFile(r, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

for i in os.walk('./data/corrected2_20180719'):
    shoout_file_names=i[2]
#    and if file_name!='.DS_store'
    shoout_file_names=[(file_name) for file_name in shoout_file_names if file_name not in['Shootout_Rules_2018_final.docx', '.DS_Store']]
    shoout_file_names=sorted(shoout_file_names)
    break
print(shoout_file_names)

# ---
# 
# ##### •	Y_Nr: a unique number for each sample
# ##### •	C_Solute: type of solute i.e. Ace, Lac, NaCl, KCl, or pure water
# ##### •	Y_conSNr: the number of the consecutive scans taken the same sample replicate (1-3).
# ##### •	C_Experiment: order number in which experiments were executed. Each solute (4) measured on 3 different days gave a total of 3x4=12 experiments, labeled E1-E12.
# ##### •	Y_Molar_concentration_mM: molar concentration in millimoles.
# ##### •	Y_Mass_concentration_g100ml: mass concentration in grams per 100 ml of pure water. 
# ##### •	Y_Room_temperature: temperature of the room at the time of scanning. 
# ##### •	Y_Room_relHumidity: relative humidity of the room at the time of scanning. 
# 
# ---

# In[279]:





# In[319]:


def load_datasets(files=shoout_file_names):
    dataset_dict={}
    headers=['Y_Nr','C_Solute', 'Y_conSNr','C_Experiment', 'Y_Molar_concentration_mM','Y_Mass_concentration_g100ml','Y_Room_temperature','Y_Room_relHumidity']
    for (file) in files:
        
        ## Remove the .txt for dict
        file_key_name=(file[:-5])
        
        ## Tab delimited files
        full_dataset=pd.read_table(f'./data/corrected2_20180719/{file}', sep='\t', lineterminator='\r')
        full_dataset=full_dataset.dropna()

        if file_key_name[:10] == 'Validation':
            headers=['Y_Nr','Y_conSNr']
            
        experimental_data=full_dataset[headers]
        spectral_data=full_dataset[full_dataset.columns[len(headers):]]
        spectral_data.columns=[re.sub('w','',wave_l) for wave_l in spectral_data.columns]
        spectral_data.insert(0, 'Y_Nr', experimental_data['Y_Nr'])
        
        
        
        dataset_dict[file_key_name]={'EXPERIMENTAL':experimental_data,'SPECTRA':spectral_data}
                
    return(dataset_dict)


# In[320]:


data_sets=load_datasets()


# In[322]:


for set_name in data_sets:
   # print(d_s)
    for data_name in data_sets[set_name]:
       # print(d_ss)
        data_sets[set_name][data_name].to_csv(f'./data/{set_name}_{data_name}.csv', index=False)



