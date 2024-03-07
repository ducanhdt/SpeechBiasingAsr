import copy
from transformers import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Config,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Attention,
    Wav2Vec2FeedForward,
    Wav2Vec2Encoder,
)
from transformers.modeling_outputs import TokenClassifierOutput, CausalLMOutput

import torch
from torch import nn
import warnings
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from address_database import AddressDatabase


_HIDDEN_STATES_START_POSITION = 2

def merge(fragments):
    if not fragments:
        return []
    merged = [fragments[0]]
    for i in range(1, len(fragments)):
        if merged[-1][0] + merged[-1][1] + 3 > fragments[i][0]:
            merged[-1] = (
                merged[-1][0],
                fragments[i][0] + fragments[i][1] - merged[-1][0],
            )
        else:
            merged.append(fragments[i])
    return merged
class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2Vec2AddressHandle(Wav2Vec2PreTrainedModel):
    def __init__(
        self, config, db_path="data/address_db.pt", num_chunk=1, top_k_search=1
    ):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        config.num_labels = 3
        self.top_k_search = top_k_search
        # token classifier
        self.dropout = nn.Dropout(config.final_dropout)
        num_layers = (
            config.num_hidden_layers + 1
        )  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        self.post_encoder_tagger = Wav2Vec2EncoderLayer(config)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

        # post encoder
        post_encode_config = copy.deepcopy(config)
        post_encode_config.hidden_size *= 1 + self.top_k_search
        # post_encode_config.num_hidden_layers = 1
        # self.post_encoder = nn.Linear(post_encode_config.hidden_size, post_encode_config.hidden_size)
        # self.post_encoder2 = nn.Linear(post_encode_config.hidden_size, post_encode_config.hidden_size)
        # self.post_encoder3 = nn.Linear(post_encode_config.hidden_size, post_encode_config.hidden_size)
        self.post_encoder = Wav2Vec2EncoderLayer(post_encode_config)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size`696 of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size
            if hasattr(config, "add_adapter") and config.add_adapter
            else config.hidden_size
        )
        # address dababase
        self.address_database = AddressDatabase(
            dim=768, num_chunk=num_chunk, model=self.wav2vec2, db_path=db_path
        )
        # self.address_database = AddressDatabase(dim=768, model=self.wav2vec2)
        # self.address_database.load()

        self.relu = nn.ReLU()
        self.norm_shape = nn.Linear(
            output_hidden_size * (1 + self.top_k_search), output_hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)
        self.post_init()
    def get_address_awe(self, query):
        # query = query.cpu().detach().numpy()
        query = query.cpu().detach().numpy()
        return self.address_database.search(query, k=self.top_k_search)

        # return query
        # return torch.zeros((1,768,top_k))

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
        fragments = merge(fragments)
        fragments = [(a,b) for a,b in fragments if b>5]  
        return fragments

    def get_similar_w2v(self, w2v_embedding, token_logit):
        r"""
        input :
            w2v_embedding (num_frame,768)
            token_logit (num_frame,)
        output:
        """
        fragments = self.get_address_fragments(token_logit)
        # if len(fragments):
        #     print(len(fragments))
        refer_embeds = [
            torch.zeros_like(w2v_embedding) for i in range(self.top_k_search)
        ]
        for start, length in fragments:
            acoustic_address_embed = w2v_embedding[start : start + length - 1, :]
            refer_address_embeds = self.get_address_awe(acoustic_address_embed)
            for refer_address_embed, refer_embed in zip(
                refer_address_embeds, refer_embeds
            ):
                if refer_address_embed.shape[0] + start > refer_embed.shape[0]:
                    refer_address_embed = refer_address_embed[
                        : refer_embed.shape[0] - start
                    ]
                refer_embed[
                    start : start + refer_address_embed.shape[0], :
                ] = torch.tensor(refer_address_embed)
        # return [torch.tensor(r) for r in refer_embeds]
        if self.top_k_search == 1:
            return torch.tensor(refer_embeds[0])

        refer_embeds = [torch.tensor(r) for r in refer_embeds]
        return torch.cat(refer_embeds, dim=1)

    def post_process(self, pred):
        for i in range(len(pred)):
            if i == 0 and pred[i] == 1:
                pred[i] = 2

            if i > 0:
                if pred[i] == 1 and pred[i - 1] == 0:
                    pred[i] = 2
                elif pred[i] == 2 and pred[i - 1] == 2:
                    pred[i] = 1
                elif pred[i] == 2 and pred[i - 1] == 1:
                    pred[i] = 1
        return pred

    def freeze_classifier(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        for param in self.post_encoder_tagger.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        tagger: Optional[torch.Tensor] = None,
        similar_embedding=None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            True if self.config.use_weighted_layer_sum else output_hidden_states
        )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # return_w2v=True
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.dropout(hidden_states)

        hidden_states_clasify = self.post_encoder_tagger(
            hidden_states,
            # attention_mask=attention_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )[0]
        # hidden_states_clasify = self.dropout(hidden_states_clasify)

        token_logits = self.classifier(hidden_states_clasify)
        token_label = torch.argmax(token_logits, dim=2)
        # x = torch.sum(token_label == 2).item()
        # if x:
        # print(x)
        # print(tagger)
        if tagger == None:
            # print("xxx")
            similar_embeding = torch.stack(
                [
                    self.get_similar_w2v(
                        hidden_states[i], self.post_process(token_label[i])
                    )
                    for i in range(token_label.shape[0])
                ]
            )
        else:
            if similar_embedding is None:
                # print("yyy")
                similar_embeding = torch.stack(
                    [
                        self.get_similar_w2v(hidden_states[i], tagger[i])
                        for i in range(token_label.shape[0])
                    ]
                )
            else:
                # print("zzz")
                # print(similar_embedding.shape)
                similar_embeding = similar_embedding

        # similar_embeding = torch.stack([self.get_similar_w2v(hidden_states[i],token_label[i]) for i in range(token_label.shape[0])])
        # print(similar_embeding)
        # similar_embeding = torch.tensor([self.get_similar_w2v(w2v_hidden_states[0],token_logits[0])])
        stacked_tensor = torch.cat((hidden_states, similar_embeding), dim=2)
        # stacked_tensor = torch.cat((hidden_states,)+(i for i in similar_embeding), dim=2)
        post_hidden_states = self.post_encoder(stacked_tensor)[0]
        # post_hidden_states = stacked_tensor
        # post_hidden_states = self.relu(self.post_encoder(post_hidden_states))
        # post_hidden_states = self.relu(self.post_encoder2(post_hidden_states))
        # post_hidden_states = self.relu(self.post_encoder3(post_hidden_states))

        hidden_states = (
            # self.relu(self.norm_shape(post_hidden_states)) + hidden_states
            self.dropout(self.relu(self.norm_shape(post_hidden_states))) + hidden_states
        )

        logits = self.lm_head(hidden_states)
        loss_ctc, loss_fct = 0, 0
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(
                    f"Label values must be <= vocab_size: {self.config.vocab_size}"
                )

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum(-1)
            ).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(
                logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss_ctc = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
        if tagger is not None:
            loss_fct = CrossEntropyLoss()(token_logits.view(-1, 3), tagger.view(-1))

        if not loss_ctc:
            loss = loss_fct
        elif not loss_fct:
            loss = loss_ctc
        else:
            # loss = loss_ctc
            # loss = 0.8 * loss_ctc + 0.2 * loss_fct
            loss = 0.2 * loss_ctc + 0.8 * loss_fct
        # print("loss_fct",loss_fct,"loss_ctc",loss_ctc)
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    import librosa

    speech_array, _ = librosa.load(
        "/home3/cuongld/ASR_team/data_ASR/vietnamese_dataset/asr-audio-header/2022/08/12/27c5fd23-a47b-4667-92d7-fe39e5cb8153_564965756-27c5fd23-a47b-4667-92d7-fe39e5cb8153.wav"
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    configuration = Wav2Vec2Config()
    configuration.num_labels = 3

    model = Wav2Vec2AddressHandle(configuration)

    inputs = feature_extractor(speech_array, return_tensors="pt", sampling_rate=16000)
    logits = model(**inputs)

    print(logits[0].shape)
