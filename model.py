import copy
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, Wav2Vec2FeatureExtractor,Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Attention, Wav2Vec2FeedForward
from transformers.modeling_outputs import TokenClassifierOutput, CausalLMOutput

import torch
from torch import nn
import warnings
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from address_database import AddressDatabase


_HIDDEN_STATES_START_POSITION = 2


class Wav2Vec2ForAudioFrameClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def freeze_custom_layers(self):
        self.wav2vec2.feature_extractor._freeze_parameters()
        self.wav2vec2.feature_projection.requires_grad_(False)
        # self.wav2vec2.encoder.requires_grad_(False)
        num_encoder_layers = len(self.wav2vec2.encoder.layers)
        for layer in self.wav2vec2.encoder.layers[:num_encoder_layers//2]:
            layer.requires_grad_(False)
            
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_w2v=None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if return_w2v:
            return (logits, hidden_states)

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
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
    def __init__(self, config):
        super().__init__(config)

        # self.token_classifer = Wav2Vec2ForAudioFrameClassification.from_pretrained(path)
        self.wav2vec2 = Wav2Vec2Model(config)
        config.num_labels=3
        # token classifier
        self.dropout = nn.Dropout(config.final_dropout)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels


        #post encoder
        post_encode_config = copy.deepcopy(config)
        post_encode_config.hidden_size*=2
        self.post_encoder = Wav2Vec2EncoderLayer(post_encode_config)
        
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size`696 of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        # address dababase
        self.address_database = AddressDatabase(dim=768,model=self.wav2vec2)
        self.address_database.load()
        
        self.relu = nn.ReLU()
        self.norm_shape = nn.Linear(output_hidden_size*2, output_hidden_size)
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)
    
    def get_address_awe(self,query, top_k=1):
        # query = query.cpu().detach().numpy()
        query = query.cpu().detach().numpy()
        return self.address_database.search(query)
    
        # return query
        # return torch.zeros((1,768,top_k))
    def get_address_fragments(self,token_logit):
        fragments = []
        current_start = None
        # longest_end = None
        longest_length = 0

        for i, val in enumerate(token_logit):
            if val == 2:
                if current_start is not None:
                    fragments.append((current_start, longest_length ))
                current_start = i
                longest_length = 1

            elif val == 1 and current_start is not None:
                # if longest_end is None or i > longest_end:
                    # longest_end = i
                longest_length += 1
            else:
                if current_start is not None:
                    fragments.append((current_start, longest_length ))
                    current_start = None
                    # longest_end = None
                    longest_length = 0

        # Check if there's a fragment ending at the last position
        if current_start is not None:
            fragments.append((current_start, longest_length ))
        return fragments
    
    def get_similar_w2v(self, w2v_embedding, token_logit):
        r'''
        input : 
            w2v_embedding (num_frame,768)
            token_logit (num_frame,)
        output:
        '''
        fragments = self.get_address_fragments(token_logit)
        refer_embed = torch.zeros_like(w2v_embedding)
        for start, length in fragments:
            acoustic_address_embed = w2v_embedding[start:start+length-1,:]
            refer_address_embed = self.get_address_awe(acoustic_address_embed)
            if refer_address_embed.shape[0] + start > refer_embed.shape[0]:
                refer_address_embed = refer_address_embed[:refer_embed.shape[0]-start]
            refer_embed[start:start+refer_address_embed.shape[0],:] = torch.tensor(refer_address_embed)
        return torch.tensor(refer_embed)
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        refer_values:Optional[torch.Tensor] = None,
        tagger:Optional[torch.Tensor] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

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
        token_logits = self.classifier(hidden_states)

        token_label = torch.argmax(token_logits, dim=2)
        
        if not tagger:
            similar_embeding = torch.stack([self.get_similar_w2v(hidden_states[i],token_label[i]) for i in range(token_label.shape[0])])
        else:
            similar_embeding = torch.stack([self.get_similar_w2v(hidden_states[i],tagger[i]) for i in range(token_label.shape[0])])
        # similar_embeding = torch.tensor([self.get_similar_w2v(w2v_hidden_states[0],token_logits[0])])
        stacked_tensor = torch.cat((hidden_states, similar_embeding), dim=2)

        post_hidden_states = self.post_encoder(stacked_tensor)[0]
        
        hidden_states = self.dropout(self.relu(self.norm_shape(post_hidden_states))) + hidden_states

        logits = self.lm_head(hidden_states) 
        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

class Wav2Vec2AddressHandle2(Wav2Vec2AddressHandle):
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        tagger:Optional[torch.Tensor] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

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
        token_logits = self.classifier(hidden_states)
        token_label = torch.argmax(token_logits, dim=2)
        if tagger == None:
            similar_embeding = torch.stack([self.get_similar_w2v(hidden_states[i],token_label[i]) for i in range(token_label.shape[0])])
        else:
            # print("xxx")
            similar_embeding = torch.stack([self.get_similar_w2v(hidden_states[i],tagger[i]) for i in range(token_label.shape[0])])
        # print(similar_embeding)
        # similar_embeding = torch.tensor([self.get_similar_w2v(w2v_hidden_states[0],token_logits[0])])
        stacked_tensor = torch.cat((hidden_states, similar_embeding), dim=2)

        post_hidden_states = self.post_encoder(stacked_tensor)[0]
        
        hidden_states = self.dropout(self.relu(self.norm_shape(post_hidden_states))) + hidden_states

        logits = self.lm_head(hidden_states) 
        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)


            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                ) 
                if tagger is not None:
                    loss_fct = CrossEntropyLoss()
                    loss += loss_fct(token_logits.view(-1, 3), tagger.view(-1))
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

if __name__=="__main__":
    import librosa
    speech_array, _ = librosa.load("/home3/cuongld/ASR_team/data_ASR/vietnamese_dataset/asr-audio-header/2022/08/12/27c5fd23-a47b-4667-92d7-fe39e5cb8153_564965756-27c5fd23-a47b-4667-92d7-fe39e5cb8153.wav")

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    configuration = Wav2Vec2Config()
    configuration.num_labels = 3

    # model = Wav2Vec2AddressHandle(configuration)
    model = Wav2Vec2ForAudioFrameClassificationLSTM(configuration)

    inputs = feature_extractor(speech_array, return_tensors="pt", sampling_rate=16000)
    logits = model(**inputs)


    print(logits[0].shape)