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

import os
import unittest
import json
import random
import shutil
import pytest

from transformers import is_torch_available

if is_torch_available():
    import torch

    from transformers import (XLNetConfig, XLNetModel, XLNetLMHeadModel, XLNetForSequenceClassification, XLNetForQuestionAnswering)
    from transformers.modeling_xlnet import XLNET_PRETRAINED_MODEL_ARCHIVE_MAP
else:
    pytestmark = pytest.mark.skip("Require Torch")

from .modeling_common_test import (CommonTestCases, ids_tensor)
from .configuration_common_test import ConfigTester

class XLNetModelTest(CommonTestCases.CommonModelTester):

    all_model_classes=(XLNetModel, XLNetLMHeadModel,
                    XLNetForSequenceClassification, XLNetForQuestionAnswering) if is_torch_available() else ()
    test_pruning = False

    class XLNetModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     mem_len=10,
                     clamp_len=-1,
                     reuse_len=15,
                     is_training=True,
                     use_labels=True,
                     vocab_size=99,
                     cutoffs=[10, 50, 80],
                     hidden_size=32,
                     num_attention_heads=4,
                     d_inner=128,
                     num_hidden_layers=5,
                     max_position_embeddings=10,
                     type_sequence_label_size=2,
                     untie_r=True,
                     bi_data=False,
                     same_length=False,
                     initializer_range=0.05,
                     seed=1,
                     type_vocab_size=2,
            ):
            """
            Initialize the embeddings.

            Args:
                self: (todo): write your description
                parent: (todo): write your description
                batch_size: (int): write your description
                seq_length: (int): write your description
                mem_len: (todo): write your description
                clamp_len: (int): write your description
                reuse_len: (bool): write your description
                is_training: (bool): write your description
                use_labels: (bool): write your description
                vocab_size: (int): write your description
                cutoffs: (float): write your description
                hidden_size: (int): write your description
                num_attention_heads: (int): write your description
                d_inner: (int): write your description
                num_hidden_layers: (int): write your description
                max_position_embeddings: (int): write your description
                type_sequence_label_size: (int): write your description
                untie_r: (todo): write your description
                bi_data: (todo): write your description
                same_length: (int): write your description
                initializer_range: (todo): write your description
                seed: (int): write your description
                type_vocab_size: (int): write your description
            """
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.mem_len = mem_len
            # self.key_len = seq_length + mem_len
            self.clamp_len = clamp_len
            self.reuse_len = reuse_len
            self.is_training = is_training
            self.use_labels = use_labels
            self.vocab_size = vocab_size
            self.cutoffs = cutoffs
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.d_inner = d_inner
            self.num_hidden_layers = num_hidden_layers
            self.max_position_embeddings = max_position_embeddings
            self.bi_data = bi_data
            self.untie_r = untie_r
            self.same_length = same_length
            self.initializer_range = initializer_range
            self.seed = seed
            self.type_vocab_size = type_vocab_size
            self.type_sequence_label_size = type_sequence_label_size

        def prepare_config_and_inputs(self):
            """
            Prepare the input_config_config_and_inputs.

            Args:
                self: (todo): write your description
            """
            input_ids_1 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            input_ids_2 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            segment_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
            input_mask = ids_tensor([self.batch_size, self.seq_length], 2).float()

            input_ids_q = ids_tensor([self.batch_size, self.seq_length + 1], self.vocab_size)
            perm_mask = torch.zeros(self.batch_size, self.seq_length + 1, self.seq_length + 1, dtype=torch.float)
            perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
            target_mapping = torch.zeros(self.batch_size, 1, self.seq_length + 1, dtype=torch.float)
            target_mapping[:, 0, -1] = 1.0  # predict last token

            sequence_labels = None
            lm_labels = None
            is_impossible_labels = None
            if self.use_labels:
                lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
                sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
                is_impossible_labels = ids_tensor([self.batch_size], 2).float()

            config = XLNetConfig(
                vocab_size_or_config_json_file=self.vocab_size,
                d_model=self.hidden_size,
                n_head=self.num_attention_heads,
                d_inner=self.d_inner,
                n_layer=self.num_hidden_layers,
                untie_r=self.untie_r,
                max_position_embeddings=self.max_position_embeddings,
                mem_len=self.mem_len,
                clamp_len=self.clamp_len,
                same_length=self.same_length,
                reuse_len=self.reuse_len,
                bi_data=self.bi_data,
                initializer_range=self.initializer_range,
                num_labels=self.type_sequence_label_size)

            return (config, input_ids_1, input_ids_2, input_ids_q, perm_mask, input_mask,
                    target_mapping, segment_ids, lm_labels, sequence_labels, is_impossible_labels)

        def set_seed(self):
            """
            Sets the seed.

            Args:
                self: (todo): write your description
            """
            random.seed(self.seed)
            torch.manual_seed(self.seed)

        def create_and_check_xlnet_base_model(self, config, input_ids_1, input_ids_2, input_ids_q, perm_mask, input_mask,
                target_mapping, segment_ids, lm_labels, sequence_labels, is_impossible_labels):
            """
            Create the model.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids_1: (str): write your description
                input_ids_2: (str): write your description
                input_ids_q: (str): write your description
                perm_mask: (todo): write your description
                input_mask: (str): write your description
                target_mapping: (str): write your description
                segment_ids: (str): write your description
                lm_labels: (todo): write your description
                sequence_labels: (todo): write your description
                is_impossible_labels: (bool): write your description
            """
            model = XLNetModel(config)
            model.eval()

            _, _ = model(input_ids_1, input_mask=input_mask)
            _, _ = model(input_ids_1, attention_mask=input_mask)
            _, _ = model(input_ids_1, token_type_ids=segment_ids)
            outputs, mems_1 = model(input_ids_1)

            result = {
                "mems_1": mems_1,
                "outputs": outputs,
            }

            config.mem_len = 0
            model = XLNetModel(config)
            model.eval()
            no_mems_outputs = model(input_ids_1)
            self.parent.assertEqual(len(no_mems_outputs), 1)

            self.parent.assertListEqual(
                list(result["outputs"].size()),
                [self.batch_size, self.seq_length, self.hidden_size])
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers)

        def create_and_check_xlnet_lm_head(self, config, input_ids_1, input_ids_2, input_ids_q, perm_mask, input_mask,
                target_mapping, segment_ids, lm_labels, sequence_labels, is_impossible_labels):
            """
            Create the network.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids_1: (str): write your description
                input_ids_2: (str): write your description
                input_ids_q: (str): write your description
                perm_mask: (todo): write your description
                input_mask: (str): write your description
                target_mapping: (str): write your description
                segment_ids: (str): write your description
                lm_labels: (int): write your description
                sequence_labels: (todo): write your description
                is_impossible_labels: (bool): write your description
            """
            model = XLNetLMHeadModel(config)
            model.eval()

            loss_1, all_logits_1, mems_1 = model(input_ids_1, token_type_ids=segment_ids, labels=lm_labels)

            loss_2, all_logits_2, mems_2 = model(input_ids_2, token_type_ids=segment_ids, labels=lm_labels, mems=mems_1)

            logits, _ = model(input_ids_q, perm_mask=perm_mask, target_mapping=target_mapping)

            result = {
                "loss_1": loss_1,
                "mems_1": mems_1,
                "all_logits_1": all_logits_1,
                "loss_2": loss_2,
                "mems_2": mems_2,
                "all_logits_2": all_logits_2,
            }

            self.parent.assertListEqual(
                list(result["loss_1"].size()),
                [])
            self.parent.assertListEqual(
                list(result["all_logits_1"].size()),
                [self.batch_size, self.seq_length, self.vocab_size])
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers)

            self.parent.assertListEqual(
                list(result["loss_2"].size()),
                [])
            self.parent.assertListEqual(
                list(result["all_logits_2"].size()),
                [self.batch_size, self.seq_length, self.vocab_size])
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_2"]),
                [[self.mem_len, self.batch_size, self.hidden_size]] * self.num_hidden_layers)

        def create_and_check_xlnet_qa(self, config, input_ids_1, input_ids_2, input_ids_q, perm_mask, input_mask,
                target_mapping, segment_ids, lm_labels, sequence_labels, is_impossible_labels):
            """
            Create the model.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids_1: (str): write your description
                input_ids_2: (str): write your description
                input_ids_q: (str): write your description
                perm_mask: (todo): write your description
                input_mask: (todo): write your description
                target_mapping: (str): write your description
                segment_ids: (str): write your description
                lm_labels: (todo): write your description
                sequence_labels: (todo): write your description
                is_impossible_labels: (bool): write your description
            """
            model = XLNetForQuestionAnswering(config)
            model.eval()

            outputs = model(input_ids_1)
            start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits, mems = outputs

            outputs = model(input_ids_1, start_positions=sequence_labels,
                                         end_positions=sequence_labels,
                                         cls_index=sequence_labels,
                                         is_impossible=is_impossible_labels,
                                         p_mask=input_mask)

            outputs = model(input_ids_1, start_positions=sequence_labels,
                                         end_positions=sequence_labels,
                                         cls_index=sequence_labels,
                                         is_impossible=is_impossible_labels)

            total_loss, mems = outputs

            outputs = model(input_ids_1, start_positions=sequence_labels,
                                         end_positions=sequence_labels)

            total_loss, mems = outputs

            result = {
                "loss": total_loss,
                "start_top_log_probs": start_top_log_probs,
                "start_top_index": start_top_index,
                "end_top_log_probs": end_top_log_probs,
                "end_top_index": end_top_index,
                "cls_logits": cls_logits,
                "mems": mems,
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
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers)

        def create_and_check_xlnet_sequence_classif(self, config, input_ids_1, input_ids_2, input_ids_q, perm_mask, input_mask,
                target_mapping, segment_ids, lm_labels, sequence_labels, is_impossible_labels):
            """
            Create the model.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids_1: (str): write your description
                input_ids_2: (str): write your description
                input_ids_q: (str): write your description
                perm_mask: (todo): write your description
                input_mask: (todo): write your description
                target_mapping: (todo): write your description
                segment_ids: (str): write your description
                lm_labels: (todo): write your description
                sequence_labels: (todo): write your description
                is_impossible_labels: (bool): write your description
            """
            model = XLNetForSequenceClassification(config)
            model.eval()

            logits, mems_1 = model(input_ids_1)
            loss, logits, mems_1 = model(input_ids_1, labels=sequence_labels)

            result = {
                "loss": loss,
                "mems_1": mems_1,
                "logits": logits,
            }

            self.parent.assertListEqual(
                list(result["loss"].size()),
                [])
            self.parent.assertListEqual(
                list(result["logits"].size()),
                [self.batch_size, self.type_sequence_label_size])
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers)

        def prepare_config_and_inputs_for_common(self):
            """
            Prepare inputs for input inputs.

            Args:
                self: (todo): write your description
            """
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids_1, input_ids_2, input_ids_q, perm_mask, input_mask,
                target_mapping, segment_ids, lm_labels,
                sequence_labels, is_impossible_labels) = config_and_inputs
            inputs_dict = {'input_ids': input_ids_1}
            return config, inputs_dict


    def setUp(self):
        """
        Sets upter model

        Args:
            self: (todo): write your description
        """
        self.model_tester = XLNetModelTest.XLNetModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XLNetConfig, d_inner=37)

    def test_config(self):
        """
        Test if test test configuration.

        Args:
            self: (todo): write your description
        """
        self.config_tester.run_common_tests()

    def test_xlnet_base_model(self):
        """
        Test if the model was created

        Args:
            self: (todo): write your description
        """
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_base_model(*config_and_inputs)

    def test_xlnet_lm_head(self):
        """
        Creates a lmnet model.

        Args:
            self: (todo): write your description
        """
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_lm_head(*config_and_inputs) 

    def test_xlnet_sequence_classif(self):
        """
        Test for the model for a model.

        Args:
            self: (todo): write your description
        """
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_sequence_classif(*config_and_inputs)

    def test_xlnet_qa(self):
        """
        Test if the model was clicked

        Args:
            self: (todo): write your description
        """
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_qa(*config_and_inputs)

    @pytest.mark.slow
    def test_model_from_pretrained(self):
        """
        Test for a pre - based on - trained.

        Args:
            self: (todo): write your description
        """
        cache_dir = "/tmp/transformers_test/"
        for model_name in list(XLNET_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = XLNetModel.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
