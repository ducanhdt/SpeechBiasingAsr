import json
import time
import os

import numpy as np
import kenlm
import librosa
import pandas as pd
import torch
import torchaudio
from datasets import load_dataset
from jiwer import wer, cer
from pyctcdecode import Alphabet
from pyctcdecode.decoder import BeamSearchDecoderCTC
from pyctcdecode.language_model import LanguageModel
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from model import Wav2Vec2AddressHandle
from rich import print

data_path = "final_data/test_aicc_add.csv"
model_path = "ASR_final/checkpoint-282"
output_path = "tmp.csv"

unigram_path = "model/common_vn_syllables.txt"
kenlm_model_path = "model/kenlm_model_qc_update_fix.binary"

db_path = "data/address_db.pt"
num_chunk = 1
top_k_search = 3
use_pretrain_processor = False


if use_pretrain_processor:
    processor = Wav2Vec2Processor.from_pretrained(model_path)
else:
    tokenizer = Wav2Vec2CTCTokenizer(
        "model/vocab_vi.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

# model = Wav2Vec2ForCTC.from_pretrained(
#     "model/average_b2_last_15_add1",
#     pad_token_id=processor.tokenizer.pad_token_id,
#     vocab_size=len(processor.tokenizer)
# )

model = Wav2Vec2AddressHandle.from_pretrained(
    model_path,
    db_path=db_path,
    num_chunk=num_chunk,
    top_k_search=top_k_search,
)

model.eval()
model = model.to("cuda")
# model.address_database.to_gpu()
with open(unigram_path) as f:
    unigram_list = [t.lower() for t in f.read().strip().split("\n")]
kenlm_model = LanguageModel(
    kenlm.Model(kenlm_model_path),
    alpha=0.8,
    beta=3.0,
    unigrams=unigram_list,
)

sorted_vocab = {k: v for k, v in sorted(
    processor.tokenizer.get_vocab().items(), key=lambda item: item[1])}

labels = list(sorted_vocab.keys())

alphabet = Alphabet.build_alphabet(labels)
decoder = BeamSearchDecoderCTC(alphabet, kenlm_model)

test_dataset = (
    load_dataset("csv", data_files=data_path,
                 split="train", cache_dir=".cache")
    .shuffle(seed=42)
    .select(range(100))
)


def read_audio(batch):
    speech_array, _ = librosa.load(batch["new_path"], sr=16000)
    batch["input_values"] = speech_array
    batch["input_length"] = len(batch["input_values"]) / 16000

    batch["tagger"] = np.array(json.loads(batch["labels"]))
    # batch['tagger'] = np.array(json.loads(batch['preds']))
    return batch


start = time.perf_counter()
print("Reading audio and sorting")
test_dataset = test_dataset.map(read_audio)
test_dataset.sort("input_length", reverse=True)


def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(
            batch["input_values"], device="cuda").unsqueeze(0)
        tagger = torch.tensor(batch["tagger"], device="cuda").unsqueeze(0)
        # logits = model(input_values,tagger=tagger)
        logits = model(input_values).logits

    # infer no lm
    # print(logits[0])
    predicted_ids = torch.argmax(logits, dim=-1)
    batch["prediction_without_lm"] = (
        processor.batch_decode(predicted_ids)[0].replace("[PAD]", "").strip()
    )

    # infer lm
    logits = logits[0].cpu().numpy()
    batch["prediction"] = decoder.decode(
        logits, beam_width=20, token_min_logp=-10, beam_prune_logp=-10
    ).strip()

    return batch


print("Infering")
test_dataset = test_dataset.map(
    map_to_result, remove_columns=["input_values", "input_length"], batch_size=32
)


def post_handle(batch):
    if batch["transcript"] == None:
        batch["transcript"] = "im lặng"
    batch["transcript"] = batch["transcript"].replace("-", "")
    batch["transcript"] = (
        batch["transcript"].replace("[PAD]", " ").replace("  ", " ").strip()
    )
    if batch["transcript"] in ["", "nan", "None"]:
        batch["transcript"] = "im lặng"

    if batch["prediction"] == None:
        batch["prediction"] = "im lặng"
    batch["prediction"] = batch["prediction"].replace("-", "")
    batch["prediction"] = (
        batch["prediction"].replace("[PAD]", " ").replace("  ", " ").strip()
    )
    if batch["prediction"] in ["", "nan", "None", "[UNK]"]:
        batch["prediction"] = "im lặng"

    if batch["prediction_without_lm"] == None:
        batch["prediction_without_lm"] = "im lặng"
    batch["prediction_without_lm"] = batch["prediction_without_lm"].replace(
        "-", "")
    batch["prediction_without_lm"] = (
        batch["prediction_without_lm"].replace(
            "[PAD]", " ").replace("  ", " ").strip()
    )
    if batch["prediction_without_lm"] in ["", "nan", "None", "[UNK]"]:
        batch["prediction_without_lm"] = "im lặng"

    return batch


print("Post-processing")
test_dataset = test_dataset.map(post_handle)
print("Infer time: ", time.perf_counter() - start)
result_dataframe = pd.DataFrame(test_dataset)


# result_dataframe = pd.read_csv('/home3/tuannd/asr-training/test_huggingface.csv')
def compute_wer(texts, preds):
    return sum([wer(text, pred) for text, pred in zip(texts, preds)]) / len(texts)


def compute_cer(texts, preds):
    return sum([cer(text, pred) for text, pred in zip(texts, preds)]) / len(texts)


print(
    "WER: ", compute_wer(
        result_dataframe["transcript"], result_dataframe["prediction"])
)
print(
    "CER: ", compute_cer(
        result_dataframe["transcript"], result_dataframe["prediction"])
)

print(
    "WER without Lm: ",
    compute_wer(
        result_dataframe["transcript"], result_dataframe["prediction_without_lm"]
    ),
)
print(
    "CER without Lm: ",
    compute_cer(
        result_dataframe["transcript"], result_dataframe["prediction_without_lm"]
    ),
)

data = result_dataframe
data["address"] = data["labels"].apply(lambda x: "1" in x)

data_filtered = data[data["address"] == 1]
print("ADDRESS:")
print(
    round(compute_wer(data_filtered["transcript"],
          data_filtered["prediction"]), 4)
    * 100
)
print(
    round(
        compute_wer(
            data_filtered["transcript"], data_filtered["prediction_without_lm"]
        ),
        4,
    )
    * 100
)

# print(round(compute_cer(data_filtered['transcript'], data_filtered['prediction']),4)*100)
# print(round(compute_cer(data_filtered['transcript'], data_filtered['prediction_without_lm']),4)*100)

print("NON ADDRESS:")
data_filtered = data[data["address"] == 0]
print(
    round(compute_wer(data_filtered["transcript"],
          data_filtered["prediction"]), 4)
    * 100
)
print(
    round(
        compute_wer(
            data_filtered["transcript"], data_filtered["prediction_without_lm"]
        ),
        4,
    )
    * 100
)

result_dataframe.to_csv(output_path, index=False)


# test set updated
# old wav2vec
# 11.15 WER
# new wav2vec
#
"""
"""
