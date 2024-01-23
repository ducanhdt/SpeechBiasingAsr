import json
import librosa
import numpy as np
import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from transformers import (
    Wav2Vec2Processor,
)

from datasets import load_dataset, load_metric


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    label_pad_token_id: int = -100
    top_k_search: int = 1

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
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
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        if "tagger" in features[0]:
            labels = [{"tagger": feature["tagger"]} for feature in features]
            # batch["tagger"] = [feature["tagger"] for feature in features]
            sequence_length = max([len(feature["tagger"]) for feature in features])

            def to_list(tensor_or_iterable):
                if isinstance(tensor_or_iterable, torch.Tensor):
                    return tensor_or_iterable.tolist()
                return list(tensor_or_iterable)

            batch["tagger"] = [
                to_list(label["tagger"])
                + [self.label_pad_token_id] * (sequence_length - len(label["tagger"]))
                for label in labels
            ]
            batch["tagger"] = torch.tensor(batch["tagger"], dtype=torch.int64)
        if "similar_embedding" in features[0]:
            similar_embedding = [
                torch.tensor(feature["similar_embedding"])
                if torch.tensor(feature["similar_embedding"]).shape[1] != 1
                else torch.zeros((int(feature["similar_embedding"][0][0]), 768))
                for feature in features
            ]
            # batch["similar_embedding"] = [feature["similar_embedding"] for feature in features]
            sequence_length = max([len(feature) for feature in similar_embedding])
            padded_embeddings = np.zeros(
                (len(features), sequence_length, self.top_k_search * 768)
            )
            for i, tensor in enumerate(similar_embedding):
                current_len = tensor.shape[0]  # Get the current length of the tensor
                if current_len <= sequence_length:
                    # If the tensor is shorter than max_len, pad it
                    padded_embeddings[i, :current_len, :] = tensor
                else:
                    # If the tensor is longer than max_len, truncate it
                    padded_embeddings[i, :, :] = tensor[:sequence_length, :]

            batch["similar_embedding"] = torch.tensor(
                padded_embeddings, dtype=torch.int64
            )
        return batch


def compute_metrics(pred, processor):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    label_str = ["im lặng" if label == "" else label for label in label_str]
    pred_str = ["im lặng" if pred == "" else pred for pred in pred_str]
    wer_metric = load_metric("wer")

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def speech_file_to_array_fn(batch, processor, model):
    speech_array, _ = librosa.load(batch["new_path"], sr=16000)
    batch["input_values"] = speech_array
    batch["input_length"] = len(batch["input_values"]) / 16000
    batch["tagger"] = np.array(json.loads(batch["labels"]))
    # batch['labels'] = batch['transcript']

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids
    if 1:
        # if 2 in batch['tagger']:
        extract_features = model.address_database.get_embed_from_file(batch["new_path"])
        # batch['similar_embedding'] = torch.tensor(len(batch['tagger']))
        # batch['similar_embedding'] = model.address_database.get_similar_w2v(
        #     torch.tensor(extract_features), batch['tagger'])
        batch["similar_embedding"] = model.get_similar_w2v(
            torch.tensor(extract_features), batch["tagger"]
        )
        del extract_features
    else:
        batch["similar_embedding"] = torch.tensor(
            [[len(batch["tagger"])]], dtype=torch.float32
        )
    return batch
