# coding=utf-8
# Copyright 2018 XXX Authors.
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
from io import open

from transformers.tokenization_bert import (XxxTokenizer, VOCAB_FILES_NAMES)

from .tokenization_tests_commons import CommonTestCases

class XxxTokenizationTest(CommonTestCases.CommonTokenizerTester):

    tokenizer_class = XxxTokenizer

    def setUp(self):
        """
        Set vocab vocabulary.

        Args:
            self: (todo): write your description
        """
        super(XxxTokenizationTest, self).setUp()

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
        return XxxTokenizer.from_pretrained(self.tmpdirname, **kwargs)

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


if __name__ == '__main__':
    unittest.main()
