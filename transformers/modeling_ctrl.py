# coding=utf-8
# Copyright 2018 Salesforce and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch CTRL model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import json
import logging
import math
import os
import sys
from io import open
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter

from .modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from .configuration_ctrl import CTRLConfig
from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)

CTRL_PRETRAINED_MODEL_ARCHIVE_MAP = {"ctrl": "https://storage.googleapis.com/sf-ctrl/pytorch/seqlen256_v1.bin"}


def angle_defn(pos, i, d_model_size):
    """
    Return the angle of the angle at pos.

    Args:
        pos: (int): write your description
        i: (todo): write your description
        d_model_size: (int): write your description
    """
    angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / d_model_size)
    return pos * angle_rates

def positional_encoding(position, d_model_size, dtype):
    """
    Positional encoding of the position.

    Args:
        position: (int): write your description
        d_model_size: (int): write your description
        dtype: (todo): write your description
    """
    # create the sinusoidal pattern for the positional encoding
    angle_rads = (angle_defn(torch.arange(position, dtype=dtype).unsqueeze(1),
                  torch.arange(d_model_size, dtype=dtype).unsqueeze(0),
                  d_model_size))

    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.cat([sines, cosines], dim=-1)
    return pos_encoding

def scaled_dot_product_attention(q, k, v, mask, attention_mask=None, head_mask=None):
    """
    Compute dot product.

    Args:
        q: (todo): write your description
        k: (todo): write your description
        v: (todo): write your description
        mask: (array): write your description
        attention_mask: (int): write your description
        head_mask: (todo): write your description
    """
    # calculate attention
    matmul_qk = torch.matmul(q, k.permute(0,1,3,2))

    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e4)

    if attention_mask is not None:
        # Apply the attention mask
        scaled_attention_logits = scaled_attention_logits + attention_mask

    attention_weights = torch.softmax(scaled_attention_logits, dim=-1) 

    # Mask heads if we want to
    if head_mask is not None:
        attention_weights = attention_weights * head_mask

    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model_size, num_heads, output_attentions=False):
        """
        Initialize the model.

        Args:
            self: (todo): write your description
            d_model_size: (int): write your description
            num_heads: (int): write your description
            output_attentions: (todo): write your description
        """
        super(MultiHeadAttention, self).__init__()
        self.output_attentions = output_attentions
        self.num_heads = num_heads
        self.d_model_size = d_model_size

        self.depth = int(d_model_size / self.num_heads)

        self.Wq = torch.nn.Linear(d_model_size, d_model_size)
        self.Wk = torch.nn.Linear(d_model_size, d_model_size)
        self.Wv = torch.nn.Linear(d_model_size, d_model_size)

        self.dense = torch.nn.Linear(d_model_size, d_model_size)

    def split_into_heads(self, x, batch_size):
        """
        Split a list of a multi - layer.

        Args:
            self: (todo): write your description
            x: (array): write your description
            batch_size: (int): write your description
        """
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute([0, 2, 1, 3])

    def forward(self, v, k, q, mask, layer_past=None, attention_mask=None, head_mask=None):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            v: (todo): write your description
            k: (todo): write your description
            q: (todo): write your description
            mask: (todo): write your description
            layer_past: (todo): write your description
            attention_mask: (todo): write your description
            head_mask: (todo): write your description
        """
        batch_size = q.shape[0]

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)
        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        present = torch.stack((k, v))

        output = scaled_dot_product_attention(q, k, v, mask, attention_mask, head_mask)
        scaled_attention = output[0].permute([0, 2, 1, 3])
        attn = output[1]
        original_size_attention = scaled_attention.reshape(batch_size, -1, self.d_model_size)
        output = self.dense(original_size_attention)

        outputs = (output, present)
        if self.output_attentions:
            outputs = outputs + (attn,)
        return outputs



def point_wise_feed_forward_network(d_model_size, dff):
    """
    Point - wise convolutional layer.

    Args:
        d_model_size: (int): write your description
        dff: (todo): write your description
    """
    return torch.nn.Sequential(torch.nn.Linear(d_model_size, dff),
                               torch.nn.ReLU(),
                               torch.nn.Linear(dff, d_model_size))


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model_size, num_heads, dff, rate=0.1, output_attentions=False):
        """
        Initialize self.

        Args:
            self: (todo): write your description
            d_model_size: (int): write your description
            num_heads: (int): write your description
            dff: (todo): write your description
            rate: (todo): write your description
            output_attentions: (todo): write your description
        """
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model_size, num_heads, output_attentions)
        self.ffn = point_wise_feed_forward_network(d_model_size, dff)

        self.layernorm1 = torch.nn.LayerNorm(d_model_size, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(d_model_size, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask, layer_past=None, attention_mask=None, head_mask=None):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            mask: (todo): write your description
            layer_past: (todo): write your description
            attention_mask: (todo): write your description
            head_mask: (todo): write your description
        """
        normed = self.layernorm1(x)
        attn_outputs = self.multi_head_attention(normed, normed, normed, mask,
                                                      layer_past=layer_past,
                                                      attention_mask=attention_mask,
                                                      head_mask=head_mask)
        attn_output = attn_outputs[0]
        attn_output = self.dropout1(attn_output)
        out1 = x + attn_output

        out2 = self.layernorm2(out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output)
        out2 = out1 + ffn_output

        outputs = (out2,) + attn_outputs[1:]
        return outputs


class CTRLPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = CTRLConfig
    pretrained_model_archive_map = CTRL_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "transformer"

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


CTRL_START_DOCSTRING = r"""    CTRL model was proposed in 
    `CTRL: A Conditional Transformer Language Model for Controllable Generation`_
    by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher.
    It's a causal (unidirectional) transformer pre-trained using language modeling on a very large
    corpus of ~140 GB of text data with the first token reserved as a control code (such as Links, Books, Wikipedia etc.).

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`CTRL: A Conditional Transformer Language Model for Controllable Generation`:
        https://www.github.com/salesforce/ctrl

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.CTRLConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

CTRL_INPUTS_DOCSTRING = r"""    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            CTRL is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.
            Indices can be obtained using :class:`transformers.CTRLTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding. The token ids which have their past given to this model 
            should not be passed as input ids as they have already been computed.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The embeddings from these tokens will be summed with the respective token embeddings.
            Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""

@add_start_docstrings("The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.",
                                            CTRL_START_DOCSTRING, CTRL_INPUTS_DOCSTRING)
class CTRLModel(CTRLPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model 
            should not be passed as input ids as they have already been computed.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = CTRLTokenizer.from_pretrained('ctrl')
        model = CTRLModel.from_pretrained('ctrl')
        input_ids = torch.tensor(tokenizer.encode("Links Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            config: (todo): write your description
        """
        super(CTRLModel, self).__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.output_past = config.output_past

        self.d_model_size = config.n_embd
        self.num_layers = config.n_layer

        self.pos_encoding = positional_encoding(config.n_positions, self.d_model_size, torch.float)

        self.w = nn.Embedding(config.vocab_size, config.n_embd)

        self.dropout = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([EncoderLayer(config.n_embd,
                                             config.n_head,
                                             config.dff,
                                             config.resid_pdrop,
                                             config.output_attentions) for _ in range(config.n_layer)])
        self.layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        """
        Returns a list of the embedded inputs.

        Args:
            self: (todo): write your description
        """
        return self.w

    def set_input_embeddings(self, new_embeddings):
        """
        Parameters ---------- new_embeddings.

        Args:
            self: (todo): write your description
            new_embeddings: (todo): write your description
        """
        self.w = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
                heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):
        """
        Forward computation. forward.

        Args:
            self: (todo): write your description
            input_ids: (str): write your description
            past: (todo): write your description
            attention_mask: (todo): write your description
            token_type_ids: (str): write your description
            position_ids: (str): write your description
            head_mask: (todo): write your description
            inputs_embeds: (todo): write your description
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
            token_type_embeds = self.w(token_type_ids)
            token_type_embeds *= np.sqrt(self.d_model_size)
        else:
            token_type_embeds = 0
        position_ids = position_ids.view(-1, input_shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.w(input_ids)
        # inputs_embeds = embedded.unsqueeze(0) if len(input_ids.shape)<2 else embedded
        seq_len = input_shape[-1]
        mask = torch.triu(torch.ones(seq_len, seq_len), 1).to(inputs_embeds.device)

        inputs_embeds *= np.sqrt(self.d_model_size)

        pos_embeds = self.pos_encoding[position_ids, :].to(inputs_embeds.device)

        hidden_states = inputs_embeds + pos_embeds + token_type_embeds

        hidden_states = self.dropout(hidden_states)

        output_shape = input_shape + (inputs_embeds.size(-1),)
        presents = ()
        all_hidden_states = ()
        all_attentions = []
        for i, (h, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)
            outputs = h(hidden_states,
                        mask,
                        layer_past=layer_past,
                        attention_mask=attention_mask,
                        head_mask=head_mask[i])
            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.layernorm(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs


@add_start_docstrings("""The CTRL Model transformer with a language modeling head on top
(linear layer with weights tied to the input embeddings). """, CTRL_START_DOCSTRING, CTRL_INPUTS_DOCSTRING)
class CTRLLMHeadModel(CTRLPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model 
            should not be passed as input ids as they have already been computed.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import CTRLTokenizer, CTRLLMHeadModel

        tokenizer = CTRLTokenizer.from_pretrained('ctrl')
        model = CTRLLMHeadModel.from_pretrained('ctrl')

        input_ids = torch.tensor(tokenizer.encode("Links Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            config: (todo): write your description
        """
        super(CTRLLMHeadModel, self).__init__(config)
        self.transformer = CTRLModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        self.init_weights()

    def get_output_embeddings(self):
        """
        : returns : class : class :.

        Args:
            self: (todo): write your description
        """
        return self.lm_head

    def forward(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input_ids: (str): write your description
            past: (todo): write your description
            attention_mask: (todo): write your description
            token_type_ids: (str): write your description
            position_ids: (str): write your description
            head_mask: (todo): write your description
            inputs_embeds: (todo): write your description
            labels: (todo): write your description
        """
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds)

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
