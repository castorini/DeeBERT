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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import shutil
import pytest
import logging

from transformers import is_tf_available

if is_tf_available():
    from transformers import (AutoConfig, BertConfig,
                                      TFAutoModel, TFBertModel,
                                      TFAutoModelWithLMHead, TFBertForMaskedLM,
                                      TFAutoModelForSequenceClassification, TFBertForSequenceClassification,
                                      TFAutoModelForQuestionAnswering, TFBertForQuestionAnswering)
    from transformers.modeling_tf_bert import TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP

    from .modeling_common_test import (CommonTestCases, ids_tensor)
    from .configuration_common_test import ConfigTester
else:
    pytestmark = pytest.mark.skip("Require TensorFlow")


class TFAutoModelTest(unittest.TestCase):
    def test_model_from_pretrained(self):
        """
        Test if the hdf5 model from the given hdf5.

        Args:
            self: (todo): write your description
        """
        import h5py
        self.assertTrue(h5py.version.hdf5_version.startswith("1.10"))

        logging.basicConfig(level=logging.INFO)
        # for model_name in list(TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        for model_name in ['bert-base-uncased']:
            config = AutoConfig.from_pretrained(model_name, force_download=True)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModel.from_pretrained(model_name, force_download=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertModel)

    def test_lmhead_model_from_pretrained(self):
        """
        Create a lmhead model is a lmhead configuration.

        Args:
            self: (todo): write your description
        """
        logging.basicConfig(level=logging.INFO)
        # for model_name in list(TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        for model_name in ['bert-base-uncased']:
            config = AutoConfig.from_pretrained(model_name, force_download=True)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelWithLMHead.from_pretrained(model_name, force_download=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForMaskedLM)

    def test_sequence_classification_model_from_pretrained(self):
        """
        Test for the classification classification.

        Args:
            self: (todo): write your description
        """
        logging.basicConfig(level=logging.INFO)
        # for model_name in list(TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        for model_name in ['bert-base-uncased']:
            config = AutoConfig.from_pretrained(model_name, force_download=True)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelForSequenceClassification.from_pretrained(model_name, force_download=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForSequenceClassification)

    def test_question_answering_model_from_pretrained(self):
        """
        Test if a question config file.

        Args:
            self: (todo): write your description
        """
        logging.basicConfig(level=logging.INFO)
        # for model_name in list(TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        for model_name in ['bert-base-uncased']:
            config = AutoConfig.from_pretrained(model_name, force_download=True)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelForQuestionAnswering.from_pretrained(model_name, force_download=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForQuestionAnswering)


if __name__ == "__main__":
    unittest.main()
