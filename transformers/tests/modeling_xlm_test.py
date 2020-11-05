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

from transformers import is_torch_available

if is_torch_available():
    from transformers import (XLMConfig, XLMModel, XLMWithLMHeadModel, XLMForQuestionAnswering,
                                      XLMForSequenceClassification, XLMForQuestionAnsweringSimple)
    from transformers.modeling_xlm import XLM_PRETRAINED_MODEL_ARCHIVE_MAP
else:
    pytestmark = pytest.mark.skip("Require Torch")

from .modeling_common_test import (CommonTestCases, ids_tensor)
from .configuration_common_test import ConfigTester


class XLMModelTest(CommonTestCases.CommonModelTester):

    all_model_classes = (XLMModel, XLMWithLMHeadModel, XLMForQuestionAnswering,
                         XLMForSequenceClassification, XLMForQuestionAnsweringSimple) if is_torch_available() else ()


    class XLMModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     is_training=True,
                     use_input_lengths=True,
                     use_token_type_ids=True,
                     use_labels=True,
                     gelu_activation=True,
                     sinusoidal_embeddings=False,
                     causal=False,
                     asm=False,
                     n_langs=2,
                     vocab_size=99,
                     n_special=0,
                     hidden_size=32,
                     num_hidden_layers=5,
                     num_attention_heads=4,
                     hidden_dropout_prob=0.1,
                     attention_probs_dropout_prob=0.1,
                     max_position_embeddings=512,
                     type_vocab_size=16,
                     type_sequence_label_size=2,
                     initializer_range=0.02,
                     num_labels=3,
                     num_choices=4,
                     summary_type="last",
                     use_proj=True,
                     scope=None,
                    ):
            """
            Initialize the model.

            Args:
                self: (todo): write your description
                parent: (todo): write your description
                batch_size: (int): write your description
                seq_length: (int): write your description
                is_training: (bool): write your description
                use_input_lengths: (int): write your description
                use_token_type_ids: (str): write your description
                use_labels: (bool): write your description
                gelu_activation: (str): write your description
                sinusoidal_embeddings: (int): write your description
                causal: (todo): write your description
                asm: (str): write your description
                n_langs: (todo): write your description
                vocab_size: (int): write your description
                n_special: (str): write your description
                hidden_size: (int): write your description
                num_hidden_layers: (int): write your description
                num_attention_heads: (int): write your description
                hidden_dropout_prob: (todo): write your description
                attention_probs_dropout_prob: (todo): write your description
                max_position_embeddings: (int): write your description
                type_vocab_size: (int): write your description
                type_sequence_label_size: (int): write your description
                initializer_range: (todo): write your description
                num_labels: (int): write your description
                num_choices: (int): write your description
                summary_type: (str): write your description
                use_proj: (bool): write your description
                scope: (str): write your description
            """
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_input_lengths = use_input_lengths
            self.use_token_type_ids = use_token_type_ids
            self.use_labels = use_labels
            self.gelu_activation = gelu_activation
            self.sinusoidal_embeddings = sinusoidal_embeddings
            self.asm = asm
            self.n_langs = n_langs
            self.vocab_size = vocab_size
            self.n_special = n_special
            self.summary_type = summary_type
            self.causal = causal
            self.use_proj = use_proj
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.n_langs = n_langs
            self.type_sequence_label_size = type_sequence_label_size
            self.initializer_range = initializer_range
            self.summary_type = summary_type
            self.num_labels = num_labels
            self.num_choices = num_choices
            self.scope = scope

        def prepare_config_and_inputs(self):
            """
            Prepare the input tensors.

            Args:
                self: (todo): write your description
            """
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            input_mask = ids_tensor([self.batch_size, self.seq_length], 2).float()

            input_lengths = None
            if self.use_input_lengths:
                input_lengths = ids_tensor([self.batch_size], vocab_size=2) + self.seq_length - 2  # small variation of seq_length

            token_type_ids = None
            if self.use_token_type_ids:
                token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.n_langs)

            sequence_labels = None
            token_labels = None
            is_impossible_labels = None
            if self.use_labels:
                sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
                token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
                is_impossible_labels = ids_tensor([self.batch_size], 2).float()

            config = XLMConfig(
                 vocab_size_or_config_json_file=self.vocab_size,
                 n_special=self.n_special,
                 emb_dim=self.hidden_size,
                 n_layers=self.num_hidden_layers,
                 n_heads=self.num_attention_heads,
                 dropout=self.hidden_dropout_prob,
                 attention_dropout=self.attention_probs_dropout_prob,
                 gelu_activation=self.gelu_activation,
                 sinusoidal_embeddings=self.sinusoidal_embeddings,
                 asm=self.asm,
                 causal=self.causal,
                 n_langs=self.n_langs,
                 max_position_embeddings=self.max_position_embeddings,
                 initializer_range=self.initializer_range,
                 summary_type=self.summary_type,
                 use_proj=self.use_proj)

            return config, input_ids, token_type_ids, input_lengths, sequence_labels, token_labels, is_impossible_labels, input_mask

        def check_loss_output(self, result):
            """
            Check that the output.

            Args:
                self: (todo): write your description
                result: (todo): write your description
            """
            self.parent.assertListEqual(
                list(result["loss"].size()),
                [])

        def create_and_check_xlm_model(self, config, input_ids, token_type_ids, input_lengths, sequence_labels, token_labels, is_impossible_labels, input_mask):
            """
            Create the model.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids: (str): write your description
                token_type_ids: (str): write your description
                input_lengths: (int): write your description
                sequence_labels: (str): write your description
                token_labels: (str): write your description
                is_impossible_labels: (todo): write your description
                input_mask: (todo): write your description
            """
            model = XLMModel(config=config)
            model.eval()
            outputs = model(input_ids, lengths=input_lengths, langs=token_type_ids)
            outputs = model(input_ids, langs=token_type_ids)
            outputs = model(input_ids)
            sequence_output = outputs[0]
            result = {
                "sequence_output": sequence_output,
            }
            self.parent.assertListEqual(
                list(result["sequence_output"].size()),
                [self.batch_size, self.seq_length, self.hidden_size])


        def create_and_check_xlm_lm_head(self, config, input_ids, token_type_ids, input_lengths, sequence_labels, token_labels, is_impossible_labels, input_mask):
            """
            Create the model and embedding.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids: (str): write your description
                token_type_ids: (str): write your description
                input_lengths: (int): write your description
                sequence_labels: (todo): write your description
                token_labels: (str): write your description
                is_impossible_labels: (bool): write your description
                input_mask: (str): write your description
            """
            model = XLMWithLMHeadModel(config)
            model.eval()

            loss, logits = model(input_ids, token_type_ids=token_type_ids, labels=token_labels)

            result = {
                "loss": loss,
                "logits": logits,
            }

            self.parent.assertListEqual(
                list(result["loss"].size()),
                [])
            self.parent.assertListEqual(
                list(result["logits"].size()),
                [self.batch_size, self.seq_length, self.vocab_size])


        def create_and_check_xlm_simple_qa(self, config, input_ids, token_type_ids, input_lengths, sequence_labels, token_labels, is_impossible_labels, input_mask):
            """
            Create a simple model.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids: (str): write your description
                token_type_ids: (str): write your description
                input_lengths: (int): write your description
                sequence_labels: (list): write your description
                token_labels: (str): write your description
                is_impossible_labels: (todo): write your description
                input_mask: (todo): write your description
            """
            model = XLMForQuestionAnsweringSimple(config)
            model.eval()

            outputs = model(input_ids)

            outputs = model(input_ids, start_positions=sequence_labels,
                                       end_positions=sequence_labels)
            loss, start_logits, end_logits = outputs

            result = {
                "loss": loss,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
            self.parent.assertListEqual(
                list(result["start_logits"].size()),
                [self.batch_size, self.seq_length])
            self.parent.assertListEqual(
                list(result["end_logits"].size()),
                [self.batch_size, self.seq_length])
            self.check_loss_output(result)


        def create_and_check_xlm_qa(self, config, input_ids, token_type_ids, input_lengths, sequence_labels, token_labels, is_impossible_labels, input_mask):
            """
            Run the model on the model.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids: (str): write your description
                token_type_ids: (str): write your description
                input_lengths: (int): write your description
                sequence_labels: (todo): write your description
                token_labels: (str): write your description
                is_impossible_labels: (todo): write your description
                input_mask: (todo): write your description
            """
            model = XLMForQuestionAnswering(config)
            model.eval()

            outputs = model(input_ids)
            start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits = outputs

            outputs = model(input_ids, start_positions=sequence_labels,
                                         end_positions=sequence_labels,
                                         cls_index=sequence_labels,
                                         is_impossible=is_impossible_labels,
                                         p_mask=input_mask)

            outputs = model(input_ids, start_positions=sequence_labels,
                                         end_positions=sequence_labels,
                                         cls_index=sequence_labels,
                                         is_impossible=is_impossible_labels)

            (total_loss,) = outputs

            outputs = model(input_ids, start_positions=sequence_labels,
                                         end_positions=sequence_labels)

            (total_loss,) = outputs

            result = {
                "loss": total_loss,
                "start_top_log_probs": start_top_log_probs,
                "start_top_index": start_top_index,
                "end_top_log_probs": end_top_log_probs,
                "end_top_index": end_top_index,
                "cls_logits": cls_logits,
            }

            self.parent.assertListEqual(
                list(result["loss"].size()),
                [])
            self.parent.assertListEqual(
                list(result["start_top_log_probs"].size()),
                [self.batch_size, model.config.start_n_top])
            self.parent.assertListEqual(
                list(result["start_top_index"].size()),
                [self.batch_size, model.config.start_n_top])
            self.parent.assertListEqual(
                list(result["end_top_log_probs"].size()),
                [self.batch_size, model.config.start_n_top * model.config.end_n_top])
            self.parent.assertListEqual(
                list(result["end_top_index"].size()),
                [self.batch_size, model.config.start_n_top * model.config.end_n_top])
            self.parent.assertListEqual(
                list(result["cls_logits"].size()),
                [self.batch_size])


        def create_and_check_xlm_sequence_classif(self, config, input_ids, token_type_ids, input_lengths, sequence_labels, token_labels, is_impossible_labels, input_mask):
            """
            Create the model for training.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids: (str): write your description
                token_type_ids: (str): write your description
                input_lengths: (int): write your description
                sequence_labels: (todo): write your description
                token_labels: (str): write your description
                is_impossible_labels: (bool): write your description
                input_mask: (todo): write your description
            """
            model = XLMForSequenceClassification(config)
            model.eval()

            (logits,) = model(input_ids)
            loss, logits = model(input_ids, labels=sequence_labels)

            result = {
                "loss": loss,
                "logits": logits,
            }

            self.parent.assertListEqual(
                list(result["loss"].size()),
                [])
            self.parent.assertListEqual(
                list(result["logits"].size()),
                [self.batch_size, self.type_sequence_label_size])


        def prepare_config_and_inputs_for_common(self):
            """
            Prepare inputs for inputs.

            Args:
                self: (todo): write your description
            """
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids, token_type_ids, input_lengths,
             sequence_labels, token_labels, is_impossible_labels, input_mask) = config_and_inputs
            inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'lengths': input_lengths}
            return config, inputs_dict

    def setUp(self):
        """
        Sets up the testerter.

        Args:
            self: (todo): write your description
        """
        self.model_tester = XLMModelTest.XLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XLMConfig, emb_dim=37)

    def test_config(self):
        """
        Test if test test configuration.

        Args:
            self: (todo): write your description
        """
        self.config_tester.run_common_tests()

    def test_xlm_model(self):
        """
        Test if xlm model.

        Args:
            self: (todo): write your description
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_model(*config_and_inputs)

    def test_xlm_lm_head(self):
        """
        Evaluate the lm model.

        Args:
            self: (todo): write your description
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_lm_head(*config_and_inputs)

    def test_xlm_simple_qa(self):
        """
        Test if xlm_xlm model.

        Args:
            self: (todo): write your description
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_simple_qa(*config_and_inputs)

    def test_xlm_qa(self):
        """
        Test if xlm model was clicked

        Args:
            self: (todo): write your description
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_qa(*config_and_inputs)

    def test_xlm_sequence_classif(self):
        """
        Test for a model classifter.

        Args:
            self: (todo): write your description
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_sequence_classif(*config_and_inputs)

    @pytest.mark.slow
    def test_model_from_pretrained(self):
        """
        Test for a model from the given model.

        Args:
            self: (todo): write your description
        """
        cache_dir = "/tmp/transformers_test/"
        for model_name in list(XLM_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = XLMModel.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
