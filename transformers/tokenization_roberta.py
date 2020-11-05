# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for RoBERTa."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import json
import logging
import os
import regex as re
from io import open

from .tokenization_gpt2 import GPT2Tokenizer

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        """
        Decorator to cache a function cache.

        Args:
        """
        return lambda func: func

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    'vocab_file': 'vocab.json',
    'merges_file': 'merges.txt',
}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
        'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
        'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
        'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-vocab.json",
        'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-vocab.json",
        'roberta-base-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
        'roberta-large-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
    },
    'merges_file':
    {
        'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
        'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
        'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-merges.txt",
        'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-merges.txt",
        'roberta-base-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
        'roberta-large-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'roberta-base': 512,
    'roberta-large': 512,
    'roberta-large-mnli': 512,
    'distilroberta-base': 512,
    'roberta-base-openai-detector': 512,
    'roberta-large-openai-detector': 512,
}


class RobertaTokenizer(GPT2Tokenizer):
    """
    RoBERTa BPE tokenizer, derived from the GPT-2 tokenizer. Peculiarities:
        - Byte-level Byte-Pair-Encoding
        - Requires a space to start the input string => the encoding methods should be called with the
          ``add_prefix_space`` flag set to ``True``.
          Otherwise, this tokenizer ``encode`` and ``decode`` method will not conserve
          the absence of a space at the beginning of a string: `tokenizer.decode(tokenizer.encode("Hello")) = " Hello"`
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, errors='replace', bos_token="<s>", eos_token="</s>", sep_token="</s>",
                 cls_token="<s>", unk_token="<unk>", pad_token='<pad>', mask_token='<mask>', **kwargs):
        """
        Initialize the vocabulary file.

        Args:
            self: (todo): write your description
            vocab_file: (str): write your description
            merges_file: (str): write your description
            errors: (str): write your description
            bos_token: (str): write your description
            eos_token: (str): write your description
            sep_token: (str): write your description
            cls_token: (str): write your description
            unk_token: (str): write your description
            pad_token: (str): write your description
            mask_token: (str): write your description
        """
        super(RobertaTokenizer, self).__init__(vocab_file=vocab_file, merges_file=merges_file, errors=errors,
                                               bos_token=bos_token, eos_token=eos_token, unk_token=unk_token,
                                               sep_token=sep_token, cls_token=cls_token, pad_token=pad_token,
                                               mask_token=mask_token, **kwargs)
        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 4  # take into account special tokens

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:
            single sequence: <s> X </s>
            pair of sequences: <s> A </s></s> B </s>
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError("You should not supply a second sequence if the provided sequence of "
                                 "ids is already formated with special tokens for the model.")
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A RoBERTa sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep) * [0] + len(token_ids_1 + sep) * [1]
