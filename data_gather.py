import json
import pandas as pd
import os

from tqdm import tqdm
from preprocess_data_tagger import get_label_length
labels = []
data = pd.read_csv("data/data_asr/validate_6h.csv")
for path in tqdm(data['path']):
    # print(path)
    os.system(f"cp {path} audio/aicc/")
    label_length, duration = get_label_length(path)
    label = [0 for i in range(label_length)]
    labels.append(json.dumps(label))

data['new_path'] = data['path'].apply(lambda x: "audio/aicc/"+x.split('/')[-1])    
data['labels'] = labels
data.to_csv("./data_ducanh/validate_6h.csv")