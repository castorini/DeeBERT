# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
import pytest
from io import open

from transformers.tokenization_bert import (BasicTokenizer,
                                                    BertTokenizer,
                                                    WordpieceTokenizer,
                                                    _is_control, _is_punctuation,
                                                    _is_whitespace, VOCAB_FILES_NAMES)

from .tokenization_tests_commons import CommonTestCases

class BertTokenizationTest(CommonTestCases.CommonTokenizerTester):

    tokenizer_class = BertTokenizer

    def setUp(self):
        """
        Set vocab vocabulary.

        Args:
            self: (todo): write your description
        """
        super(BertTokenizationTest, self).setUp()

        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing", ",", "low", "lowest",
        ]
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        with open(self.vocab_file, "w", encoding='utf-8') as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_tokenizer(self, **kwargs):
        """
        Return a tokenizer.

        Args:
            self: (todo): write your description
        """
        return BertTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self):
        """
        Returns a list of outputs_text output_text can be used to generate_inputs.

        Args:
            self: (todo): write your description
        """
        input_text = u"UNwant\u00E9d,running"
        output_text = u"unwanted, running"
        return input_text, output_text

    def test_full_tokenizer(self):
        """
        Reads a list of the corpus.

        Args:
            self: (todo): write your description
        """
        tokenizer = self.tokenizer_class(self.vocab_file)

        tokens = tokenizer.tokenize(u"UNwant\u00E9d,running")
        self.assertListEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

    def test_chinese(self):
        """
        Run the test tokenizer.

        Args:
            self: (todo): write your description
        """
        tokenizer = BasicTokenizer()

        self.assertListEqual(
            tokenizer.tokenize(u"ah\u535A\u63A8zz"),
            [u"ah", u"\u535A", u"\u63A8", u"zz"])

    def test_basic_tokenizer_lower(self):
        """
        Tokenizes the tokenizer.

        Args:
            self: (todo): write your description
        """
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
            ["hello", "!", "how", "are", "you", "?"])
        self.assertListEqual(tokenizer.tokenize(u"H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):
        """
        Tokenizes tokenizer.

        Args:
            self: (todo): write your description
        """
        tokenizer = BasicTokenizer(do_lower_case=False)

        self.assertListEqual(
            tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
            ["HeLLo", "!", "how", "Are", "yoU", "?"])

    def test_wordpiece_tokenizer(self):
        """
        Returns a list of tokens.

        Args:
            self: (todo): write your description
        """
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing"
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = WordpieceTokenizer(vocab=vocab, unk_token="[UNK]")

        self.assertListEqual(tokenizer.tokenize(""), [])

        self.assertListEqual(
            tokenizer.tokenize("unwanted running"),
            ["un", "##want", "##ed", "runn", "##ing"])

        self.assertListEqual(
            tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])

    def test_is_whitespace(self):
        """
        Check if a valid whitespace.

        Args:
            self: (todo): write your description
        """
        self.assertTrue(_is_whitespace(u" "))
        self.assertTrue(_is_whitespace(u"\t"))
        self.assertTrue(_is_whitespace(u"\r"))
        self.assertTrue(_is_whitespace(u"\n"))
        self.assertTrue(_is_whitespace(u"\u00A0"))

        self.assertFalse(_is_whitespace(u"A"))
        self.assertFalse(_is_whitespace(u"-"))

    def test_is_control(self):
        """
        Set the control flag.

        Args:
            self: (todo): write your description
        """
        self.assertTrue(_is_control(u"\u0005"))

        self.assertFalse(_is_control(u"A"))
        self.assertFalse(_is_control(u" "))
        self.assertFalse(_is_control(u"\t"))
        self.assertFalse(_is_control(u"\r"))

    def test_is_punctuation(self):
        """
        Set the punctuation is punctuation.

        Args:
            self: (todo): write your description
        """
        self.assertTrue(_is_punctuation(u"-"))
        self.assertTrue(_is_punctuation(u"$"))
        self.assertTrue(_is_punctuation(u"`"))
        self.assertTrue(_is_punctuation(u"."))

        self.assertFalse(_is_punctuation(u"A"))
        self.assertFalse(_is_punctuation(u" "))

    @pytest.mark.slow
    def test_sequence_builders(self):
        """
        Test if the input tokenizer.

        Args:
            self: (todo): write your description
        """
        tokenizer = self.tokenizer_class.from_pretrained("bert-base-uncased")

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [101] + text + [102]
        assert encoded_pair == [101] + text + [102] + text_2 + [102]

if __name__ == '__main__':
    unittest.main()
