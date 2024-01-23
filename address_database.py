import os
import numpy as np
import faiss
import torchaudio
import torch
from tqdm import tqdm
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)
import pickle

import pandas as pd


class AddressDatabase:
    def __init__(
        self,
        dim=512,
        num_chunk=1,
        data_path=None,
        db_path=None,
        model=None,
        model_path="model/average_b2_last_15_add1",
    ) -> None:
        self.dim = dim
        self.num_chunk = num_chunk
        tokenizer = Wav2Vec2CTCTokenizer(f"model/vocab_vi.json")
        if not model:
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=True,
            )
            processor = Wav2Vec2Processor(
                feature_extractor=feature_extractor, tokenizer=tokenizer
            )

            self.model = Wav2Vec2Model.from_pretrained(
                model_path,
                pad_token_id=processor.tokenizer.pad_token_id,
                vocab_size=len(processor.tokenizer),
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
        else:
            self.model = model
        if db_path:
            self.load(db_path)
        elif data_path:
            self.build(data_path)

    def _encode(self, query):
        r"""
        convert w2v embedding to latent space
        """
        chunks = np.array_split(query, self.num_chunk)
        averages = [np.mean(chunk, axis=0) for chunk in chunks]

        return np.hstack(averages)

    def get_embed_from_file(self, file_path):
        # try:
        audio, _ = torchaudio.load(file_path)
        sr = 16000
        audio = torchaudio.functional.resample(audio, _, sr)
        # extract_features = self.model.wav2vec2.feature_extractor(audio)
        extract_features = self.model(audio.to(self.model.device)).last_hidden_state
        # print(extract_features.shape)
        del audio
        return extract_features[0].cpu().detach().numpy()

    def _encode_file(self, file_path):
        return self._encode(self.get_embed_from_file(file_path))

    def search(self, embed, k=1, return_file=False):
        query = self._encode(embed)
        D, I = self.index.search(np.array([query]), k)
        del query
        file_list = self.mapping["file"].iloc[I[0]].to_list()
        if return_file:
            return self.get_embed_from_file(file_list[0]), file_list
        return [self.get_embed_from_file(file) for file in file_list]

    def build(self, path="audio/address_audio"):
        # file_list = []
        self.index = faiss.IndexFlatL2(self.dim * self.num_chunk)
        embeds = []
        file_list = [f"{path}/{file}" for file in os.listdir(path)]
        # file_list = [i for i in file_list if "NGOCHUYEN" in i]
        # embeds.append(self._encode_file(file_list[0]))
        # self.index.add(np.array(embeds))
        for file in tqdm(file_list, desc="Building"):
            # file = f"{path}/{file}"
            # file_list.append(file)
            embeds.append(self._encode_file(file))
        self.mapping = pd.DataFrame({"file": file_list})
        self.index.add(np.array(embeds))
        if self.model.device == "cuda":
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print(f"load address database from {path}, using GPU")
        else:
            print(f"load address database from {path}, using CPU")

    def add(self, path="audio/address_audio"):
        # file_list = []
        embeds = []
        file_list = [f"{path}/{file}" for file in os.listdir(path)]
        # file_list = [i for i in file_list if "NGOCHUYEN" in i]
        # embeds.append(self._encode_file(file_list[0]))
        # self.index.add(np.array(embeds))
        for file in tqdm(file_list, desc="Building"):
            # file = f"{path}/{file}"
            # file_list.append(file)
            embeds.append(self._encode_file(file))
        file_list = list(self.mapping["file"]) + file_list
        self.mapping = pd.DataFrame({"file": file_list})
        self.index.add(np.array(embeds))

    def load(self, path="data/address_db.pt"):
        with open(path, "rb") as f:
            self.index, self.mapping = pickle.load(f)
            print(self.mapping)
        if self.model.device == "cuda":
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print(f"load address database from {path}, using GPU")
        else:
            print(f"load address database from {path}, using CPU")

    def to_gpu(self):
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print(f"Address database using GPU")
        except:
            pass

    def save(self, path="data/address_db.pt"):
        with open(path, "wb") as f:
            pickle.dump((self.index, self.mapping), f)
        print(f"save address database to {path}")

    def get_size(self):
        return self.index.ntotal

    def get_similar_w2v(self, w2v_embedding, token_logit, top_k=1):
        r"""
        input :
            w2v_embedding (num_frame,768)
            token_logit (num_frame,)
        output:
        """
        fragments = self.get_address_fragments(token_logit)
        refer_embed = torch.zeros_like(w2v_embedding)
        for start, length in fragments:
            acoustic_address_embed = w2v_embedding[start : start + length - 1, :]
            refer_address_embed = self.get_address_awe(acoustic_address_embed)
            if refer_address_embed.shape[0] + start > refer_embed.shape[0]:
                refer_address_embed = refer_address_embed[
                    : refer_embed.shape[0] - start
                ]
            refer_embed[start : start + refer_address_embed.shape[0], :] = torch.tensor(
                refer_address_embed
            )
        return torch.tensor(refer_embed)

    def get_address_awe(self, query, top_k=1):
        # query = query.cpu().detach().numpy()
        query = query.cpu().detach().numpy()
        return self.search(query)[0]

    def get_address_fragments(self, token_logit):
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


if __name__ == "__main__":
    # address_db = AddressDatabase(dim=768,data_path="address_audio")
    # address_db.save()

    # address_db = AddressDatabase(dim=768,data_path="audio/english_address_audio/audio",model_path="facebook/wav2vec2-base-960h")
    address_db = AddressDatabase(
        dim=768,
        db_path="data/address_english_db.pt",
        model_path="facebook/wav2vec2-base-960h",
    )
    address_db = AddressDatabase(
        dim=768,
        db_path="data/address_english_db.pt",
        model_path="facebook/wav2vec2-base-960h",
    )
    address_db.add("./audio/english_address_audio_test/")
    # address_db.load()
    # address_db.mapping['file'] = address_db.mapping['file'].apply(lambda x:f"audio/{x}")
    print(address_db.mapping)
    address_db.save("data/address_english_db_test.pt")
    # address_db.mapping['file'] = address_db.mapping['file'].apply(lambda file:f"address_audio/{file}" )

    # print("database size: ",address_db.get_size())
    # # sample_query = address_db.get_embed_from_file("address_audio/Làng_Nhì_MAIPHUONG-HN.wav")
    # sample_query = address_db.get_embed_from_file('address_audio/La_Gi_MAIPHUONG-HN.wav')
    # print(sample_query.shape)
    # res = address_db.search(sample_query,k=5,return_file=True)
    # print(res)
