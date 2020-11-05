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
import sys

from .modeling_tf_common_test import (TFCommonTestCases, ids_tensor)
from .configuration_common_test import ConfigTester

from transformers import CTRLConfig, is_tf_available

if is_tf_available():
    import tensorflow as tf
    from transformers.modeling_tf_ctrl import (TFCTRLModel, TFCTRLLMHeadModel,
                                                TF_CTRL_PRETRAINED_MODEL_ARCHIVE_MAP)
else:
    pytestmark = pytest.mark.skip("Require TensorFlow")


class TFCTRLModelTest(TFCommonTestCases.TFCommonModelTester):

    all_model_classes = (TFCTRLModel, TFCTRLLMHeadModel) if is_tf_available() else ()

    class TFCTRLModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     is_training=True,
                     use_token_type_ids=True,
                     use_input_mask=True,
                     use_labels=True,
                     use_mc_token_ids=True,
                     vocab_size=99,
                     hidden_size=32,
                     num_hidden_layers=5,
                     num_attention_heads=4,
                     intermediate_size=37,
                     hidden_act="gelu",
                     hidden_dropout_prob=0.1,
                     attention_probs_dropout_prob=0.1,
                     max_position_embeddings=512,
                     type_vocab_size=16,
                     type_sequence_label_size=2,
                     initializer_range=0.02,
                     num_labels=3,
                     num_choices=4,
                     scope=None,
                     ):
            """
            Initialize the encoder.

            Args:
                self: (todo): write your description
                parent: (todo): write your description
                batch_size: (int): write your description
                seq_length: (int): write your description
                is_training: (bool): write your description
                use_token_type_ids: (str): write your description
                use_input_mask: (bool): write your description
                use_labels: (bool): write your description
                use_mc_token_ids: (str): write your description
                vocab_size: (int): write your description
                hidden_size: (int): write your description
                num_hidden_layers: (int): write your description
                num_attention_heads: (int): write your description
                intermediate_size: (int): write your description
                hidden_act: (todo): write your description
                hidden_dropout_prob: (todo): write your description
                attention_probs_dropout_prob: (todo): write your description
                max_position_embeddings: (int): write your description
                type_vocab_size: (int): write your description
                type_sequence_label_size: (int): write your description
                initializer_range: (todo): write your description
                num_labels: (int): write your description
                num_choices: (int): write your description
                scope: (str): write your description
            """
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_token_type_ids = use_token_type_ids
            self.use_input_mask = use_input_mask
            self.use_labels = use_labels
            self.use_mc_token_ids = use_mc_token_ids
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.type_sequence_label_size = type_sequence_label_size
            self.initializer_range = initializer_range
            self.num_labels = num_labels
            self.num_choices = num_choices
            self.scope = scope

        def prepare_config_and_inputs(self):
            """
            Prepare the embeddings.

            Args:
                self: (todo): write your description
            """
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            input_mask = None
            if self.use_input_mask:
                input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            token_type_ids = None
            if self.use_token_type_ids:
                token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

            mc_token_ids = None
            if self.use_mc_token_ids:
                mc_token_ids = ids_tensor([self.batch_size, self.num_choices], self.seq_length)

            sequence_labels = None
            token_labels = None
            choice_labels = None
            if self.use_labels:
                sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
                token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
                choice_labels = ids_tensor([self.batch_size], self.num_choices)

            config = CTRLConfig(
                vocab_size_or_config_json_file=self.vocab_size,
                n_embd=self.hidden_size,
                n_layer=self.num_hidden_layers,
                n_head=self.num_attention_heads,
                # intermediate_size=self.intermediate_size,
                # hidden_act=self.hidden_act,
                # hidden_dropout_prob=self.hidden_dropout_prob,
                # attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                n_positions=self.max_position_embeddings,
                n_ctx=self.max_position_embeddings
                # type_vocab_size=self.type_vocab_size,
                # initializer_range=self.initializer_range
            )

            head_mask = ids_tensor([self.num_hidden_layers, self.num_attention_heads], 2)

            return config, input_ids, input_mask, head_mask, token_type_ids, mc_token_ids, sequence_labels, token_labels, choice_labels

        def create_and_check_ctrl_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
            """
            Create the model.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids: (str): write your description
                input_mask: (todo): write your description
                head_mask: (todo): write your description
                token_type_ids: (str): write your description
            """
            model = TFCTRLModel(config=config)
            inputs = {'input_ids': input_ids,
                      'attention_mask': input_mask,
                      'token_type_ids': token_type_ids}
            sequence_output = model(inputs)[0]

            inputs = [input_ids, None, input_mask]  # None is the input for 'past'
            sequence_output = model(inputs)[0]

            sequence_output = model(input_ids)[0]

            result = {
                "sequence_output": sequence_output.numpy(),
            }
            self.parent.assertListEqual(
                list(result["sequence_output"].shape),
                [self.batch_size, self.seq_length, self.hidden_size])


        def create_and_check_ctrl_lm_head(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
            """
            Create the lm lm lm lm for training.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids: (str): write your description
                input_mask: (str): write your description
                head_mask: (todo): write your description
                token_type_ids: (str): write your description
            """
            model = TFCTRLLMHeadModel(config=config)
            inputs = {'input_ids': input_ids,
                      'attention_mask': input_mask,
                      'token_type_ids': token_type_ids}
            prediction_scores = model(inputs)[0]
            result = {
                "prediction_scores": prediction_scores.numpy(),
            }
            self.parent.assertListEqual(
                list(result["prediction_scores"].shape),
                [self.batch_size, self.seq_length, self.vocab_size])

        def prepare_config_and_inputs_for_common(self):
            """
            Prepare inputs for inputs.

            Args:
                self: (todo): write your description
            """
            config_and_inputs = self.prepare_config_and_inputs()

            (config, input_ids, input_mask, head_mask, token_type_ids,
             mc_token_ids, sequence_labels, token_labels, choice_labels) = config_and_inputs

            inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': input_mask}
            return config, inputs_dict

    def setUp(self):
        """
        Sets the model configuration

        Args:
            self: (todo): write your description
        """
        self.model_tester = TFCTRLModelTest.TFCTRLModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CTRLConfig, n_embd=37)

    def test_config(self):
        """
        Test if test test configuration.

        Args:
            self: (todo): write your description
        """
        self.config_tester.run_common_tests()

    def test_ctrl_model(self):
        """
        Test if the model was clicked

        Args:
            self: (todo): write your description
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_ctrl_model(*config_and_inputs)

    def test_ctrl_lm_head(self):
        """
        Vel factory for lm head_lm_inputs

        Args:
            self: (todo): write your description
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_ctrl_lm_head(*config_and_inputs)

    @pytest.mark.slow
    def test_model_from_pretrained(self):
        """
        Initialize a pre - based on - disk.

        Args:
            self: (todo): write your description
        """
        cache_dir = "/tmp/transformers_test/"
        for model_name in list(TF_CTRL_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = TFCTRLModel.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()

