# coding=utf-8
# Copyright 2010, XXX authors
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
""" XXX model configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
import six
from io import open

from .configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

XXX_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'xxx-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/xxx-base-uncased-config.json",
    'xxx-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/xxx-large-uncased-config.json",
}


class XxxConfig(PretrainedConfig):
    r"""
        :class:`~transformers.XxxConfig` is the configuration class to store the configuration of a
        `XxxModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `XxxModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `XxxModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = XXX_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=50257,
                 n_positions=1024,
                 n_ctx=1024,
                 n_embd=768,
                 n_layer=12,
                 n_head=12,
                 resid_pdrop=0.1,
                 embd_pdrop=0.1,
                 attn_pdrop=0.1,
                 layer_norm_epsilon=1e-5,
                 initializer_range=0.02,

                 num_labels=1,
                 summary_type='cls_index',
                 summary_use_proj=True,
                 summary_activation=None,
                 summary_proj_to_labels=True,
                 summary_first_dropout=0.1,
                 **kwargs):
        """
        Initialize the configuration.

        Args:
            self: (todo): write your description
            vocab_size_or_config_json_file: (str): write your description
            n_positions: (int): write your description
            n_ctx: (int): write your description
            n_embd: (int): write your description
            n_layer: (todo): write your description
            n_head: (int): write your description
            resid_pdrop: (todo): write your description
            embd_pdrop: (todo): write your description
            attn_pdrop: (int): write your description
            layer_norm_epsilon: (int): write your description
            initializer_range: (todo): write your description
            num_labels: (int): write your description
            summary_type: (str): write your description
            summary_use_proj: (bool): write your description
            summary_activation: (todo): write your description
            summary_proj_to_labels: (str): write your description
            summary_first_dropout: (str): write your description
        """
        super(XxxConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size_or_config_json_file if isinstance(vocab_size_or_config_json_file, six.string_types) else -1
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        self.num_labels = num_labels
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        if isinstance(vocab_size_or_config_json_file, six.string_types):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif not isinstance(vocab_size_or_config_json_file, int):
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @property
    def max_position_embeddings(self):
        """
        The maximum positions of the maximum positions.

        Args:
            self: (todo): write your description
        """
        return self.n_positions

    @property
    def hidden_size(self):
        """
        The number of the hidden hidden size.

        Args:
            self: (todo): write your description
        """
        return self.n_embd

    @property
    def num_attention_heads(self):
        """
        The number of available attention.

        Args:
            self: (todo): write your description
        """
        return self.n_head

    @property
    def num_hidden_layers(self):
        """
        Return the number of hidden hidden layers.

        Args:
            self: (todo): write your description
        """
        return self.n_layer
