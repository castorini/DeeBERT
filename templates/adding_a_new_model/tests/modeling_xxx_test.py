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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import shutil
import pytest

from transformers import is_torch_available

from .modeling_common_test import (CommonTestCases, ids_tensor)
from .configuration_common_test import ConfigTester

if is_torch_available():
    from transformers import (XxxConfig, XxxModel, XxxForMaskedLM,
                                        XxxForNextSentencePrediction, XxxForPreTraining,
                                        XxxForQuestionAnswering, XxxForSequenceClassification,
                                        XxxForTokenClassification, XxxForMultipleChoice)
    from transformers.modeling_xxx import XXX_PRETRAINED_MODEL_ARCHIVE_MAP
else:
    pytestmark = pytest.mark.skip("Require Torch")


class XxxModelTest(CommonTestCases.CommonModelTester):

    all_model_classes = (XxxModel, XxxForMaskedLM, XxxForQuestionAnswering,
                         XxxForSequenceClassification,
                         XxxForTokenClassification) if is_torch_available() else ()

    class XxxModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     is_training=True,
                     use_input_mask=True,
                     use_token_type_ids=True,
                     use_labels=True,
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
            Initialize the model.

            Args:
                self: (todo): write your description
                parent: (todo): write your description
                batch_size: (int): write your description
                seq_length: (int): write your description
                is_training: (bool): write your description
                use_input_mask: (bool): write your description
                use_token_type_ids: (str): write your description
                use_labels: (bool): write your description
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
            self.use_input_mask = use_input_mask
            self.use_token_type_ids = use_token_type_ids
            self.use_labels = use_labels
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
            Prepare the model for training.

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

            sequence_labels = None
            token_labels = None
            choice_labels = None
            if self.use_labels:
                sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
                token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
                choice_labels = ids_tensor([self.batch_size], self.num_choices)

            config = XxxConfig(
                vocab_size_or_config_json_file=self.vocab_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
                type_vocab_size=self.type_vocab_size,
                initializer_range=self.initializer_range)

            return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

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

        def create_and_check_xxx_model(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            """
            Create the model.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids: (str): write your description
                token_type_ids: (str): write your description
                input_mask: (todo): write your description
                sequence_labels: (todo): write your description
                token_labels: (str): write your description
                choice_labels: (str): write your description
            """
            model = XxxModel(config=config)
            model.eval()
            sequence_output, pooled_output = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
            sequence_output, pooled_output = model(input_ids, token_type_ids=token_type_ids)
            sequence_output, pooled_output = model(input_ids)

            result = {
                "sequence_output": sequence_output,
                "pooled_output": pooled_output,
            }
            self.parent.assertListEqual(
                list(result["sequence_output"].size()),
                [self.batch_size, self.seq_length, self.hidden_size])
            self.parent.assertListEqual(list(result["pooled_output"].size()), [self.batch_size, self.hidden_size])


        def create_and_check_xxx_for_masked_lm(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            """
            Create a batch of targets.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids: (str): write your description
                token_type_ids: (str): write your description
                input_mask: (todo): write your description
                sequence_labels: (todo): write your description
                token_labels: (str): write your description
                choice_labels: (todo): write your description
            """
            model = XxxForMaskedLM(config=config)
            model.eval()
            loss, prediction_scores = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, masked_lm_labels=token_labels)
            result = {
                "loss": loss,
                "prediction_scores": prediction_scores,
            }
            self.parent.assertListEqual(
                list(result["prediction_scores"].size()),
                [self.batch_size, self.seq_length, self.vocab_size])
            self.check_loss_output(result)


        def create_and_check_xxx_for_question_answering(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            """
            Create a batch of examples for a batch.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids: (str): write your description
                token_type_ids: (str): write your description
                input_mask: (todo): write your description
                sequence_labels: (todo): write your description
                token_labels: (str): write your description
                choice_labels: (todo): write your description
            """
            model = XxxForQuestionAnswering(config=config)
            model.eval()
            loss, start_logits, end_logits = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids,
                                                   start_positions=sequence_labels, end_positions=sequence_labels)
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


        def create_and_check_xxx_for_sequence_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            """
            Create the classification classification.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids: (str): write your description
                token_type_ids: (str): write your description
                input_mask: (todo): write your description
                sequence_labels: (todo): write your description
                token_labels: (str): write your description
                choice_labels: (todo): write your description
            """
            config.num_labels = self.num_labels
            model = XxxForSequenceClassification(config)
            model.eval()
            loss, logits = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
            result = {
                "loss": loss,
                "logits": logits,
            }
            self.parent.assertListEqual(
                list(result["logits"].size()),
                [self.batch_size, self.num_labels])
            self.check_loss_output(result)


        def create_and_check_xxx_for_token_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            """
            Create the classification for the given token.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids: (str): write your description
                token_type_ids: (str): write your description
                input_mask: (todo): write your description
                sequence_labels: (todo): write your description
                token_labels: (str): write your description
                choice_labels: (str): write your description
            """
            config.num_labels = self.num_labels
            model = XxxForTokenClassification(config=config)
            model.eval()
            loss, logits = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
            result = {
                "loss": loss,
                "logits": logits,
            }
            self.parent.assertListEqual(
                list(result["logits"].size()),
                [self.batch_size, self.seq_length, self.num_labels])
            self.check_loss_output(result)


        def prepare_config_and_inputs_for_common(self):
            """
            Prepare inputs for inputs.

            Args:
                self: (todo): write your description
            """
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids, token_type_ids, input_mask,
             sequence_labels, token_labels, choice_labels) = config_and_inputs
            inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': input_mask}
            return config, inputs_dict

    def setUp(self):
        """
        Sets the model_tester.

        Args:
            self: (todo): write your description
        """
        self.model_tester = XxxModelTest.XxxModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XxxConfig, hidden_size=37)

    def test_config(self):
        """
        Test if test test configuration.

        Args:
            self: (todo): write your description
        """
        self.config_tester.run_common_tests()

    def test_xxx_model(self):
        """
        Test if the model was clicked.

        Args:
            self: (todo): write your description
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xxx_model(*config_and_inputs)

    def test_for_masked_lm(self):
        """
        Test for lm masked inputs.

        Args:
            self: (todo): write your description
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xxx_for_masked_lm(*config_and_inputs)

    def test_for_question_answering(self):
        """
        Test for question question question

        Args:
            self: (todo): write your description
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xxx_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        """
        Create a classification sequence classification.

        Args:
            self: (todo): write your description
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xxx_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        """
        Test for all possible.

        Args:
            self: (todo): write your description
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xxx_for_token_classification(*config_and_inputs)

    @pytest.mark.slow
    def test_model_from_pretrained(self):
        """
        Test if the model from a pre - trained.

        Args:
            self: (todo): write your description
        """
        cache_dir = "/tmp/transformers_test/"
        for model_name in list(XXX_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = XxxModel.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()
