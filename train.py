from utils import DataCollatorCTCWithPadding, speech_file_to_array_fn, compute_metrics
import json
import os
import numpy as np
import torch
import torchaudio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
)
from model import Wav2Vec2AddressHandle
from datasets import load_dataset, load_metric
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import TrainingArguments
from transformers import Trainer
import librosa
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Load parameters from external JSON file
with open("config/training_config.json", "r") as f:
    params = json.load(f)

device = params["device"]
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = params["os_environment_variable"]

train_path = params["train_path"]
test_path = params["test_path"]

trainset = load_dataset("csv", data_files=train_path, split="train", cache_dir=".cache")
testset = load_dataset("csv", data_files=test_path, split="train", cache_dir=".cache")

pretrain_model = params["pretrain_model"]
address_database = params["address_database"]
audio_search = params["audio_search"]
model: Wav2Vec2AddressHandle = Wav2Vec2AddressHandle.from_pretrained(
    pretrain_model,
    db_path=address_database,
    num_chunk=audio_search["num_chunk"],
    top_k_search=audio_search["top_k_search"],
)
model.config.mask_time_prob = 0
model.to(device)
model.address_database.to_gpu()

tokenizer_params = params["tokenizer_parameters"]
if params['use_pretrain_processor']:
    processor = Wav2Vec2Processor.from_pretrained(pretrain_model)
else:
    tokenizer = Wav2Vec2CTCTokenizer(
        tokenizer_params["vocab_file_path"],
        unk_token=tokenizer_params["unk_token"],
        pad_token=tokenizer_params["pad_token"],
        word_delimiter_token=tokenizer_params["word_delimiter_token"],
    )

    feature_extractor_params = params["feature_extractor_parameters"]
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=feature_extractor_params["feature_size"],
        sampling_rate=feature_extractor_params["sampling_rate"],
        padding_value=feature_extractor_params["padding_value"],
        do_normalize=feature_extractor_params["do_normalize"],
        return_attention_mask=feature_extractor_params["return_attention_mask"],
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )


testset = testset.map(
    lambda batch: speech_file_to_array_fn(batch, processor, model),
    new_fingerprint="dev",
)
trainset = trainset.map(
    lambda batch: speech_file_to_array_fn(batch, processor, model),
    new_fingerprint="train",
)


for param in model.wav2vec2.parameters():
    param.requires_grad = False
# model.freeze_classifier()
# model.wav2vec2.freeze_feature_encoder()
data_collator = DataCollatorCTCWithPadding(
    processor=processor, padding=True, top_k_search=audio_search["top_k_search"]
)

training_args_params = params["training_arguments"]
args = TrainingArguments(
    training_args_params["output_dir"],
    evaluation_strategy=training_args_params["evaluation_strategy"],
    save_strategy=training_args_params["save_strategy"],
    learning_rate=training_args_params["learning_rate"],
    num_train_epochs=training_args_params["num_train_epochs"],
    weight_decay=training_args_params["weight_decay"],
    push_to_hub=training_args_params["push_to_hub"],
    logging_steps=training_args_params["logging_steps"],
    report_to=training_args_params["report_to"],
    per_device_train_batch_size=training_args_params["per_device_train_batch_size"],
    gradient_accumulation_steps=training_args_params["gradient_accumulation_steps"],
    per_gpu_eval_batch_size=training_args_params["per_gpu_eval_batch_size"],
    dataloader_num_workers=training_args_params["dataloader_num_workers"],
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=trainset,
    eval_dataset=testset,
    # eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=lambda pred: compute_metrics(pred, processor),
    tokenizer=processor.feature_extractor,
)
# trainer.
# a = trainer.evaluate(ignore_keys=['tagger'])
# print(a)
# trainer.train(
# "ctc_20_model/checkpoint-68140",
#     ignore_keys_for_eval=['tagger'],
# )
# trainer.train(ignore_keys_for_eval=['tagger'], resume_from_checkpoint=True)
trainer.train(ignore_keys_for_eval=["tagger"])
