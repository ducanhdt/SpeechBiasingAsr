import json
import librosa
import numpy as np
from address_database import AddressDatabase
from datasets import load_dataset, load_metric
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
)

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
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# address_db = AddressDatabase(dim=768*3,num_chunk=3,db_path="data/address_english_db_test.pt",model_path="facebook/wav2vec2-base-960h")
# address_db = AddressDatabase(dim=768*5,data_path="audio/english_address_audio/audio/",model_path="facebook/wav2vec2-base-960h")
address_db = AddressDatabase(
    dim=768, num_chunk=3, model_path="facebook/wav2vec2-base-960h"
)
# address_db = AddressDatabase(dim=768,num_chunk=3)

# address_db = AddressDatabase(dim=512,model_path="AWE_triplet_loss/english_model/epoch1")
# address_db.load("data/address_db_Xepoch_1.pt")
# address_db.to_gpu()
# address_db.build()
# address_db.load("data/address_db.pt")
address_db.load("data/address_db_chunk3.pt")
address_db.to_gpu()
# address_db.add("audio/english_address_audio_test")
# address_db.save("data/address_db_chunk5.pt")

# test_path = 'final_data/train_with_additional_address.csv'
# train_path = 'final_data/train_aicc_tts_add.csv'
# test_path = 'final_data/train_aicc_tts_add.csv'
# test_path = 'final_data/dev_aicc_add.csv'
# test_path = 'final_data/test_aicc_add.csv'
test_path = "final_data/librispeech_test_other_2.csv"
# test_path = 'final_data/librispeech_train_tts_other500.csv'
# test_path = 'final_data/librispeech_dev_other_2.csv'


testset = load_dataset("csv", data_files=test_path, split="train", cache_dir=".cache")


def get_address_fragments(token_logit):
    fragments = []
    current_start = None
    # longest_end = None
    longest_length = 0

    for i, val in enumerate(token_logit):
        if val == 2:
            if current_start is not None:
                fragments.append((current_start, longest_length))
            current_start = i
            longest_length = 1

        elif val == 1 and current_start is not None:
            # if longest_end is None or i > longest_end:
            # longest_end = i
            longest_length += 1
        else:
            if current_start is not None:
                fragments.append((current_start, longest_length))
                current_start = None
                # longest_end = None
                longest_length = 0

    # Check if there's a fragment ending at the last position
    if current_start is not None:
        fragments.append((current_start, longest_length))
    return fragments


# count = 0


def speech_file_to_array_fn(batch):
    # speech_array, _ = librosa.load(batch["new_path"], sr=16000)
    # try:
    if 1:
        token_logit = np.array(json.loads(batch["labels"]))
        if sum(token_logit) > 0:
            w2v_embedding = address_db.get_embed_from_file(batch["new_path"])
            fragments = get_address_fragments(token_logit)
            tmp = []
            for start, length in fragments:
                if length > 1:
                    # print(length)
                    acoustic_address_embed = w2v_embedding[
                        start : start + length - 1, :
                    ]
                    _, file_list = address_db.search(
                        acoustic_address_embed, k=2, return_file=True
                    )
                    tmp.append(file_list)
                else:
                    # count+=1
                    # print(batch)
                    pass
            top1_right = 0
            topk_right = 0
            for top_k in tmp:
                check = 0
                top_k_f = [
                    " ".join(word.split("/")[-1].split("_")[:-2]).lower()
                    for word in top_k
                ]
                check_topk = int(
                    any([word_f in batch["transcript"].lower() for word_f in top_k_f])
                )
                check_top1 = int(top_k_f[0] in batch["transcript"].lower())
                # print(word,word_f)
                top1_right += check_top1
                topk_right += check_topk
            batch["total"] = len(tmp)
            batch["top_1_right"] = top1_right
            batch["top_k_right"] = topk_right
        else:
            batch["top_1_right"] = 0
            batch["top_k_right"] = 0
            batch["total"] = 0
        return batch
    # except:
    #     print(batch)
    #     batch['top_1_right'] = 0
    #     batch['top_k_right'] = 0
    #     batch['total'] = 0
    #     return(batch)
    # batch['labels'] = batch['transcript']

    # with processor.as_target_processor():
    #     batch["labels"] = processor(batch["transcript"]).input_ids
    # return batch


testset = testset.map(speech_file_to_array_fn)
testset.to_csv("final_data/tmp.csv")
top1 = np.sum(testset["top_1_right"])
top3 = np.sum(testset["top_k_right"])
total = np.sum(testset["total"])
print(f"Top 1: {top1/total} ({top1}/{total})")
print(f"Top 3: {top3/total} ({top3}/{total})")
