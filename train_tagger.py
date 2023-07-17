
import argparse
from typing import Dict, List, Optional, Union
import json 

import pandas as pd
import numpy as np
import torch
import librosa
import evaluate
from datasets import load_dataset
from dataclasses import dataclass
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2CTCTokenizer,Wav2Vec2Processor
from transformers import TrainingArguments, Trainer

# from transformers import Wav2Vec2ForAudioFrameClassification
from model import Wav2Vec2ForAudioFrameClassification

def read_audio(batch):
    speech_array, _ = librosa.load(batch["path"], sr=16000)
    batch["input_values"] = speech_array
    batch["input_length"] = len(batch["input_values"]) / 16000
    batch['labels'] = np.array(json.loads(batch['labels']))
    return batch

address_label = ['O', 'I-Add', 'B-Add']
id2label = {i: label for i, label in enumerate(address_label)}
label2id = {v: k for k, v in id2label.items()}

tokenizer = Wav2Vec2CTCTokenizer("data/vocab_vi.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        labels = [{"labels": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        sequence_length = max([len(feature["labels"]) for feature in features])

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)
        label_name = "labels"
        batch["labels"] = [
            to_list(label["labels"]) + [self.label_pad_token_id] * (sequence_length - len(label["labels"])) for label in labels
        ]
        # batch["labels"] = [
        #     to_list(label["labels"]) + [self.label_pad_token_id] * (sequence_length - len(label["labels"])) for label in labels
        # ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        # return batch

        # replace padding with -100 to ignore loss correctly
        # labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # batch["labels"] = labels

        return batch
    
data_collator = DataCollatorCTCWithPadding(processor)

def post_process(pred):
    for i in range(len(pred)):
        if i == 0 and pred[i] == 'I-Add':
            pred[i] = 'B-Add'

        if i > 0:
            if pred[i] == 'I-Add' and pred[i-1] == 'O':
                pred[i] = 'B-Add'
            elif pred[i] == 'B-Add' and pred[i-1] == 'B-Add':
                pred[i] = 'I-Add'
            elif pred[i] == 'O' and pred[i-1] == 'B-Add':
                pred[i] = 'I-Add'
    return pred

def get_segment(label):
    count = 0
    flag = False 
    begin = end = 0
    hyp = []
    for i in range(len(label)):
        if label[i] == 'B-Add':
            count += 1
            flag = True
            begin = i
            end = i
        if label[i] == 'I-Add':
            if flag:
                end = i
        if label[i] == 'O':
            if flag:
                flag = False
                hyp.append((str(count), begin*1.0, end*1.0))
    if flag:
        hyp.append((str(count), begin*1.0, end*1.0))
    return hyp

import simpleder
def get_error_rate(predictions, references):
    der = []
    pp_der = []
    for label, pred in zip(references, predictions):
        ref = get_segment(label)
        hyp = get_segment(pred)
        post_process_hyp = get_segment(post_process(pred))
        if not ref and not hyp:
            error = 0
            pp_error = 0
        elif not ref and hyp:
            error = 1
            pp_error = 1
        else:
            error = simpleder.DER(ref, hyp)
            pp_error = simpleder.DER(ref, post_process_hyp)
        der.append(error)
        pp_der.append(pp_error)
    return sum(der)/len(der), sum(pp_der)/len(pp_der)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[address_label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [address_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # print('true_labels', true_labels)
    # print('true_predictions', true_predictions)
    error_rate, pp_error_rate = get_error_rate(predictions=true_predictions, references=true_labels)
    return {
        "error_rate": error_rate,
        "pp_error_rate": pp_error_rate,
    }


if __name__ == '__main__':
    model_path = '/home4/tuannd/vbee-asr/asr-validation/pretrain-aicc'
    # model_path = '/home3/tuannd/asr-training/models/average_b2_last_15_add1'

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_kfold", default=False, type=bool, help="do kfold")
    args = parser.parse_args()
    do_kfold = args.do_kfold

    data_path = '/home4/tuannd/address_handler/tagger_data_mfa_tts.csv'
    # data_path = '/home4/tuannd/address_handler/data/mix_train_tagger.csv'
    data = pd.read_csv(data_path)
    # test_data_path = 'data/test_tagger.csv'
    if do_kfold:
        from sklearn.model_selection import KFold   
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(kf.split(data)):
            model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
                model_path,
                id2label=id2label,
                label2id=label2id,
            )

            # model.freeze_base_model()
            model.freeze_feature_encoder()
            # model.freeze_custom_layers(num_train_layers=6)
            print(f'fold {fold}')
            # train, test = data.iloc[train_index], data.iloc[test_index]
            train_data_path = f'data/tmp/train_tagger_fold{fold}.csv'
            dev_data_path = f'data/tmp/dev_tagger_fold{fold}.csv'

            # train.to_csv(train_data_path, index=False)
            # test.to_csv(dev_data_path, index=False)
            train_set = load_dataset("csv", data_files=train_data_path, split="train", cache_dir='.cache')
            dev_set = load_dataset("csv", data_files=dev_data_path, split="train", cache_dir='.cache')

            train_set = train_set.map(read_audio)
            train_set.sort("input_length", reverse=True)

            dev_set = dev_set.map(read_audio)
            dev_set.sort("input_length", reverse=True)
            args = TrainingArguments(
                f"tagger_checkpoints_asr_finetuned_freeze_feature_encoder_lstm_fold{fold}",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                per_device_train_batch_size=8,
                gradient_accumulation_steps=2,
                dataloader_num_workers=6,
                learning_rate=1e-5,
                num_train_epochs=50,
                weight_decay=0.01,
                push_to_hub=False,
                save_total_limit=10,
                report_to='tensorboard',
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_set,
                eval_dataset=dev_set,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                tokenizer=feature_extractor,
            )

            trainer.train()
    else:
        model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
            model_path,
            id2label=id2label,
            label2id=label2id,
        )

        # model.freeze_base_model()
        model.freeze_feature_encoder()
        # model.freeze_custom_layers(num_train_layers=6)

        train_set = load_dataset("csv", data_files=data_path, split="train", cache_dir='.cache')

        train_set = train_set.map(read_audio)
        train_set.sort("input_length", reverse=True)

        args = TrainingArguments(
            f"tagger_checkpoints_pretrained_freeze_feature_encoder_tts",
            # evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            dataloader_num_workers=6,
            learning_rate=1e-5,
            num_train_epochs=50,
            weight_decay=0.01,
            push_to_hub=False,
            save_total_limit=10,
            report_to='tensorboard',
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_set,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=feature_extractor,
        )

        trainer.train()
# trainer.train(resume_from_checkpoint=True)

