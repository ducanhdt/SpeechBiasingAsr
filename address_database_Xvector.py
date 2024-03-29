import os
import numpy as np
import faiss
import torchaudio
import torch
from tqdm import tqdm
from transformers import (
    Wav2Vec2ForXVector,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)
import pickle

import pandas as pd


class AddressDatabase:
    def __init__(
        self, dim=512, data_path=None, db_path=None, model=None, model_path=None
    ) -> None:
        self.dim = dim
        if not model_path:
            model_path = "AWE_triplet_loss/trained_model_freeze_base_1e-4/epoch0"

        # model_path = '/home3/tuannd/asr-training/models/checkpoint-deployed-old'
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

            self.model = Wav2Vec2ForXVector.from_pretrained(
                model_path,
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
        return (
            self.model.encode_emb(torch.tensor([query], device=self.model.device))[0]
            .cpu()
            .detach()
            .numpy()
        )

    def get_embed_from_file(self, file_path):
        audio, _ = torchaudio.load(file_path)
        sr = 16000
        audio = torchaudio.functional.resample(audio, _, sr)
        # extract_features = self.model.wav2vec2.feature_extractor(audio)
        extract_features = self.model.get_hidden_states(audio.to(self.model.device))
        # print(extract_features.shape)
        # print(extract_features.shape)
        del audio
        return extract_features[0].cpu().detach().numpy()

    def _encode_file(self, file_path):
        audio, _ = torchaudio.load(file_path)
        sr = 16000
        audio = torchaudio.functional.resample(audio, _, sr)
        # extract_features = self.model.wav2vec2.feature_extractor(audio)
        extract_features = self.model(audio.to(self.model.device)).embeddings
        # print(extract_features.shape)
        del audio
        return extract_features[0].cpu().detach().numpy()
        # return self._encode(self.get_embed_from_file(file_path))

    def search(self, embed, k=1, return_file=False):
        query = self._encode(embed)
        D, I = self.index.search(np.array([query]), k)
        del query
        file_list = self.mapping["file"].iloc[I[0]].to_list()
        if return_file:
            return self.get_embed_from_file(file_list[0]), file_list
        return self.get_embed_from_file(file_list[0])

    def build(self, path="audio/address_audio"):
        # file_list = []
        embeds = []
        file_list = [f"{path}/{file}" for file in os.listdir(path)]
        # file_list = [i for i in file_list if "NGOCHUYEN" in i]
        for file in tqdm(file_list, desc="Building"):
            embeds.append(self._encode_file(file))
            # file = f"{path}/{file}"
            # file_list.append(file)
        self.mapping = pd.DataFrame({"file": file_list})
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(np.array(embeds))
        if self.model.device == "cuda":
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print(f"load address database from {path}, using GPU")
        else:
            print(f"load address database from {path}, using CPU")

    def load(self, path="data/address_db_X.pt"):
        with open(path, "rb") as f:
            self.index, self.mapping = pickle.load(f)
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

    def save(self, path="data/address_db_X.pt"):
        with open(path, "wb") as f:
            pickle.dump((self.index, self.mapping), f)
        print(f"save address database to {path}")

    def get_size(self):
        return self.index.ntotal


if __name__ == "__main__":
    address_db = AddressDatabase(dim=512)

    print("database size: ", address_db.get_size())
    sample_query = address_db.get_embed_from_file(
        "audio/address_audio/La_Gi_MAIPHUONG-HN.wav"
    )
    print(sample_query.shape)
    res = address_db.search(sample_query, k=5, return_file=True)
    print(res)
