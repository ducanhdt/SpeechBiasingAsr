import json
import os
import torchaudio

import torch 
from datasets import load_dataset
import pandas as pd
import numpy as np 
from rich import print
from model import Wav2Vec2ForAudioFrameClassification


def read_audio(batch):
    audio, _ = torchaudio.load(batch["path"])
    sr = 16000
    audio = torchaudio.functional.resample(audio, _, sr)
    batch["input_values"] = audio 
    batch["input_length"] = len(batch["input_values"]) / 16000
    batch['labels'] = np.array(json.loads(batch['labels']))
    return batch

def post_process(pred):
    for i in range(len(pred)):
        if i == 0 and pred[i] == 1:
            pred[i] = 2

        if i > 0:
            if pred[i] == 1 and pred[i-1] == 0:
                pred[i] = 2
            elif pred[i] == 2 and pred[i-1] == 2:
                pred[i] = 1
            elif pred[i] == 0 and pred[i-1] == 2:
                pred[i] = 1
    return pred

def get_segment(label):
    count = 0
    flag = False 
    begin = end = 0
    hyp = []
    for i in range(len(label)):
        if label[i] == 2:
            count += 1
            flag = True
            begin = i
            end = i
        if label[i] == 1:
            if flag:
                end = i
        if label[i] == 0:
            if flag:
                flag = False
                hyp.append((str(count), begin*1.0, end*1.0))
    if flag:
        hyp.append((str(count), begin*1.0, end*1.0))
    return hyp

import simpleder
def calculate_error(ref, hyp):
    if not ref and not hyp:
        error = 0
    elif not ref and hyp:
        error = 1
    else:
        error = simpleder.DER(ref, hyp)
    return error 

def infer(batch):
    with torch.no_grad():
        input_values = torch.tensor(
            batch["input_values"], device="cuda")
        output = model(input_values)
        batch["pred"] = output[0].argmax(dim=2).cpu().numpy().tolist()[0]
        batch["der"] = calculate_error(get_segment(batch["labels"]), get_segment(batch["pred"]))
        batch["pred"] = post_process(batch["pred"])
        batch["pp_der"] = calculate_error(get_segment(batch["labels"]), get_segment(batch["pred"]))
    batch["labels"] = np.array(batch["labels"])
    return batch

do_kfold = True
if do_kfold:
    mode = "freeze_feature_encoder"
    print("Mode: ", mode)
    for i in range(5):
        cpkts = f"/home4/tuannd/address_handler/tagger_checkpoints_asr_finetuned_{mode}_lstm_fold{i}"
        model_path = os.path.join(cpkts, os.listdir(cpkts)[-1])
        print("Model path: ", model_path)
        model = Wav2Vec2ForAudioFrameClassification.from_pretrained(model_path)
        model.to("cuda")
        test_data_path = "data/test_tagger.csv"
        # evaluate on test set
        test_set = load_dataset("csv", data_files=test_data_path, split="train", cache_dir='.cache')
        test_set = test_set.map(read_audio)
        test_set = test_set.map(infer, remove_columns=["input_values", "input_length"], batch_size=16)


        result = pd.DataFrame(test_set)
        print('Fold: ', i)
        print('DER: ', result['der'].mean())
        print('PP-DER: ', result['pp_der'].mean())
else:
    # model_path = '/home4/tuannd/address_handler/tagger_checkpoints_pretrained_freeze_feature_encoder_mix_train/checkpoint-89460'
    model_path = '/home4/tuannd/address_handler/tagger_checkpoints_asr_finetuned_freeze_feature_encoder_lstm_fold0/checkpoint-6450'
    model = Wav2Vec2ForAudioFrameClassification.from_pretrained(model_path)
    model = model.to("cuda")
    test_data_path = "data/test_tagger.csv"
    # evaluate on test set
    test_set = load_dataset("csv", data_files=test_data_path, split="train", cache_dir='.cache')
    test_set = test_set.map(read_audio)
    test_set = test_set.map(infer, remove_columns=["input_values", "input_length"], batch_size=16)
    result = pd.DataFrame(test_set)
    print('DER: ', result['der'].mean())
    print('PP-DER: ', result['pp_der'].mean())

# result.to_csv("data/test_tagger_pred.csv", index=False)
# i = 0
# audio_path = test_set.iloc[i]['path']
# gt = json.loads(test_set.iloc[i]['labels'])

# audio_input, _ = torchaudio.load(audio_path)

# output = model(audio_input)
# # print(output[0].shape)
# print(output[0].argmax(dim=2).numpy())
# print(gt)


# print(classification_report(gt, output[0].argmax(dim=2).numpy().tolist()[0], target_names=address_label)) 
# output to labels
# labels = output[0].argmax(dim=2).numpy().tolist()[0]
# print(labels)
# print(gt)



