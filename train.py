
import json
import os
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2CTCTokenizer,Wav2Vec2Processor,Wav2Vec2Config, Wav2Vec2ForCTC
from model import Wav2Vec2ForAudioFrameClassification, Wav2Vec2AddressHandle2
from datasets import load_dataset, load_metric
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import TrainingArguments
from transformers import Trainer
import librosa 
 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import warnings
warnings.filterwarnings('ignore')

train_path = 'data_ducanh/train_aicc_tts_add.csv'
test_path = 'data_ducanh/test_aicc_add.csv'

trainset = load_dataset(
    "csv", data_files=train_path, split="train", cache_dir='.cache')
testset = load_dataset(
    "csv", data_files=test_path, split="train", cache_dir='.cache')
# model_path = '/home4/tuannd/vbee-asr/asr-validation/saved_checkpoints_b2_sampled_small_label_smoothing_v2/checkpoint-25170'
model_path = '/home3/tuannd/asr-training/models/average_b2_last_15_add1'

tokenizer = Wav2Vec2CTCTokenizer("data/vocab_vi.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def speech_file_to_array_fn(batch):
    speech_array, _ = librosa.load(batch["path"], sr=16000)
    batch["input_values"] = speech_array
    batch["input_length"] = len(batch["input_values"]) / 16000
    batch['tagger'] = np.array(json.loads(batch['labels']))
    # batch['labels'] = batch['transcript']

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids
    return batch

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
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        labels = [{"tagger": feature["tagger"]} for feature in features]
        # batch["tagger"] = [feature["tagger"] for feature in features]
        sequence_length = max([len(feature["tagger"]) for feature in features])
        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)
        batch["tagger"] = [
            to_list(label["tagger"]) + [self.label_pad_token_id] * (sequence_length - len(label["tagger"])) for label in labels
        ]
        batch["tagger"] = torch.tensor(batch["tagger"], dtype=torch.int64)
        
        return batch


testset = testset.map(speech_file_to_array_fn)
trainset = trainset.map(speech_file_to_array_fn)
# testset = testset.map(prepare_dataset)

# testset.sort("duration", reverse=True)

# address_label = ['O', 'I-Add', 'B-Add']
# id2label = {i: label for i, label in enumerate(address_label)}
# label2id = {v: k for k, v in id2label.items()}


# Load model
# model = Wav2Vec2ForCTC.from_pretrained(
#     model_path
# )
model = Wav2Vec2AddressHandle2.from_pretrained(
    model_path
)
# configuration = Wav2Vec2Config()
# configuration.num_labels = 3
# configuration.vocab_size=97
# a = torch.load(model_path+"/pytorch_model.bin")


# model = Wav2Vec2AddressHandle(configuration)
# # model.freeze_feature_extractor()
# model.token_classifer.wav2vec2.load_state_dict({
# k.replace('wav2vec2.',''):a[k] for k in a if 'wav2vec2.'in k
# })
# model = Wav2Vec2ForCTC(configuration)
# model.wav2vec2.load_state_dict({
# k.replace('wav2vec2.',''):a[k] for k in a if 'wav2vec2.'in k
# })
# model.lm_head.load_state_dict({
# k.replace('lm_head.',''):a[k] for k in a if 'lm_head.'in k
# })
# model.load_state_dict(a)


# model.lm_head.load_state_dict({
#     "weight":a['lm_head.weight'],
#     "bias":a['lm_head.bias']
# })

# for param in model.lm_head.parameters():
#     param.requires_grad = False

# for param in model.token_classifer.parameters():
#     param.requires_grad = False

for param in model.wav2vec2.parameters():
    param.requires_grad = False
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    label_str = ['im lặng' if label == '' else label for label in label_str]
    pred_str = ['im lặng' if pred == '' else pred for pred in pred_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

args = TrainingArguments(
    "output_address_mix_tts_aicc",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=20,
    weight_decay=0.01,
    push_to_hub=False,
    # report_to='none',
    logging_steps=100,
    report_to='tensorboard',
    per_device_train_batch_size=8,
    per_gpu_eval_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=1000,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=trainset,
    eval_dataset=testset,
    # eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)

trainer.train(ignore_keys_for_eval=['tagger'])
# trainer.train(resume_from_checkpoint=True)

