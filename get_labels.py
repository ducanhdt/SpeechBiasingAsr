import math
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import torchaudio

import torch 
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
model_path = '/home3/tuannd/asr-training/models/average_b2_last_15_add1'
# model_path = '/home3/tuannd/asr-training/models/checkpoint-deployed-old'
tokenizer = Wav2Vec2CTCTokenizer(f'/home3/tuannd/asr-training/data/vocab_vi.json')

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=tokenizer)

model = Wav2Vec2ForCTC.from_pretrained(
    model_path,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)
model.to("cuda")

def get_label_length(audio_path):
    audio, _ = torchaudio.load(audio_path)
    sr = 16000
    audio = audio.to("cuda")
    audio = torchaudio.functional.resample(audio, _, sr)
    extract_features = model.wav2vec2.feature_extractor(audio)
    duration = audio.shape[-1] / sr
    return extract_features.shape[-1], duration

# print(get_label_length('/home4/tuannd/ASR_team/data_ASR/dcm/63f2f8911fdb3eb1ea7f7fa4/63f2f8b785bb563ac8dd4c81_0_3500.wav'))
import json
import pandas as pd
# data = pd.read_csv("cut_results_address.csv")

# labels = []
# paths = []
# trans = []
# for path in data['original_path'].unique():
#     sample = data[data["original_path"]==path]
#     label_length, duration = get_label_length(path)
#     label = [0 for i in range(label_length)]

#     second_per_frame = duration/label_length
#     for index, row in sample.iterrows():
#         start = math.floor(row['start']/second_per_frame)
#         end = math.ceil(row['end']/second_per_frame)
#         label[start] = 2
#         for i in range(start+1,end):
#             label[i] = 1
#         tran = row['transcript']
#     labels.append(json.dumps(label))
#     paths.append(path)
#     trans.append(tran)

# pd.DataFrame({"path":data['original_path'].unique(),"labels":labels}).to_csv("tagger_data_v2.csv", index=False)

with open('/home4/tuannd/address_handler/data/tagger_data_tts.json', 'r') as f:
    data = json.load(f)

labels = []
paths = []
entities = []
for path in tqdm(data):
    label_length, duration = get_label_length(path)
    address = []
    label = [0 for i in range(label_length)]
    second_per_frame = duration/label_length
    for entity in data[path]:
        start = math.floor(entity['start']/second_per_frame)
        
        end = math.ceil(entity['end']/second_per_frame)
        if entity['end'] >= duration:
            end = label_length
        label[start] = 2
        for i in range(start+1,end):
            # print(path, i, label_length)
            label[i] = 1 
        address.append(entity['address'])
    labels.append(json.dumps(label))
    paths.append(path)
    entities.append(json.dumps(address, ensure_ascii=False))

pd.DataFrame({"path":paths,"labels":labels, "entities":entities}).to_csv("tagger_data_mfa_tts.csv", index=False)