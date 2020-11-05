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
import random
import shutil
import pytest

from .modeling_tf_common_test import (TFCommonTestCases, ids_tensor)
from .configuration_common_test import ConfigTester

from transformers import TransfoXLConfig, is_tf_available

if is_tf_available():
    import tensorflow as tf
    from transformers.modeling_tf_transfo_xl import (TFTransfoXLModel,
                                                             TFTransfoXLLMHeadModel,
                                                             TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP)
else:
    pytestmark = pytest.mark.skip("Require TensorFlow")


class TFTransfoXLModelTest(TFCommonTestCases.TFCommonModelTester):

    all_model_classes = (TFTransfoXLModel, TFTransfoXLLMHeadModel) if is_tf_available() else ()
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False

    class TFTransfoXLModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     mem_len=30,
                     clamp_len=15,
                     is_training=True,
                     use_labels=True,
                     vocab_size=99,
                     cutoffs=[10, 50, 80],
                     hidden_size=32,
                     d_embed=32,
                     num_attention_heads=4,
                     d_head=8,
                     d_inner=128,
                     div_val=2,
                     num_hidden_layers=5,
                     scope=None,
                     seed=1,
                     ):
            """
            Initialize a batch.

            Args:
                self: (todo): write your description
                parent: (todo): write your description
                batch_size: (int): write your description
                seq_length: (int): write your description
                mem_len: (todo): write your description
                clamp_len: (int): write your description
                is_training: (bool): write your description
                use_labels: (bool): write your description
                vocab_size: (int): write your description
                cutoffs: (float): write your description
                hidden_size: (int): write your description
                d_embed: (int): write your description
                num_attention_heads: (int): write your description
                d_head: (todo): write your description
                d_inner: (int): write your description
                div_val: (todo): write your description
                num_hidden_layers: (int): write your description
                scope: (str): write your description
                seed: (int): write your description
            """
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.mem_len = mem_len
            self.key_len = seq_length + mem_len
            self.clamp_len = clamp_len
            self.is_training = is_training
            self.use_labels = use_labels
            self.vocab_size = vocab_size
            self.cutoffs = cutoffs
            self.hidden_size = hidden_size
            self.d_embed = d_embed
            self.num_attention_heads = num_attention_heads
            self.d_head = d_head
            self.d_inner = d_inner
            self.div_val = div_val
            self.num_hidden_layers = num_hidden_layers
            self.scope = scope
            self.seed = seed

        def prepare_config_and_inputs(self):
            """
            Prepare the model and target and target.

            Args:
                self: (todo): write your description
            """
            input_ids_1 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            input_ids_2 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            lm_labels = None
            if self.use_labels:
                lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            config = TransfoXLConfig(
                vocab_size_or_config_json_file=self.vocab_size,
                mem_len=self.mem_len,
                clamp_len=self.clamp_len,
                cutoffs=self.cutoffs,
                d_model=self.hidden_size,
                d_embed=self.d_embed,
                n_head=self.num_attention_heads,
                d_head=self.d_head,
                d_inner=self.d_inner,
                div_val=self.div_val,
                n_layer=self.num_hidden_layers)

            return (config, input_ids_1, input_ids_2, lm_labels)

        def set_seed(self):
            """
            Sets the seed.

            Args:
                self: (todo): write your description
            """
            random.seed(self.seed)
            tf.random.set_seed(self.seed)

        def create_and_check_transfo_xl_model(self, config, input_ids_1, input_ids_2, lm_labels):
            """
            Create the model.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids_1: (str): write your description
                input_ids_2: (str): write your description
                lm_labels: (str): write your description
            """
            model = TFTransfoXLModel(config)

            hidden_states_1, mems_1 = model(input_ids_1)

            inputs = {'input_ids': input_ids_2,
                      'mems': mems_1}

            hidden_states_2, mems_2 = model(inputs)

            result = {
                "hidden_states_1": hidden_states_1.numpy(),
                "mems_1": [mem.numpy() for mem in mems_1],
                "hidden_states_2": hidden_states_2.numpy(),
                "mems_2": [mem.numpy() for mem in mems_2],
            }

            self.parent.assertListEqual(
                list(result["hidden_states_1"].shape),
                [self.batch_size, self.seq_length, self.hidden_size])
            self.parent.assertListEqual(
                list(result["hidden_states_2"].shape),
                [self.batch_size, self.seq_length, self.hidden_size])
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems_1"]),
                [[self.mem_len, self.batch_size, self.hidden_size]] * self.num_hidden_layers)
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems_2"]),
                [[self.mem_len, self.batch_size, self.hidden_size]] * self.num_hidden_layers)


        def create_and_check_transfo_xl_lm_head(self, config, input_ids_1, input_ids_2, lm_labels):
            """
            Create the lm model.

            Args:
                self: (todo): write your description
                config: (todo): write your description
                input_ids_1: (str): write your description
                input_ids_2: (str): write your description
                lm_labels: (int): write your description
            """
            model = TFTransfoXLLMHeadModel(config)

            lm_logits_1, mems_1 = model(input_ids_1)

            inputs = {'input_ids': input_ids_1,
                      'labels': lm_labels}
            _, mems_1 = model(inputs)

            lm_logits_2, mems_2 = model([input_ids_2, mems_1])

            inputs = {'input_ids': input_ids_1,
                      'mems': mems_1,
                      'labels': lm_labels}

            _, mems_2 = model(inputs)

            result = {
                "mems_1": [mem.numpy() for mem in mems_1],
                "lm_logits_1": lm_logits_1.numpy(),
                "mems_2": [mem.numpy() for mem in mems_2],
                "lm_logits_2": lm_logits_2.numpy(),
            }

            self.parent.assertListEqual(
                list(result["lm_logits_1"].shape),
                [self.batch_size, self.seq_length, self.vocab_size])
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems_1"]),
                [[self.mem_len, self.batch_size, self.hidden_size]] * self.num_hidden_layers)

            self.parent.assertListEqual(
                list(result["lm_logits_2"].shape),
                [self.batch_size, self.seq_length, self.vocab_size])
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems_2"]),
                [[self.mem_len, self.batch_size, self.hidden_size]] * self.num_hidden_layers)

        def prepare_config_and_inputs_for_common(self):
            """
            Prepare inputs for config inputs.

            Args:
                self: (todo): write your description
            """
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids_1, input_ids_2, lm_labels) = config_and_inputs
            inputs_dict = {'input_ids': input_ids_1}
            return config, inputs_dict


    def setUp(self):
        """
        Set the transfoXL model

        Args:
            self: (todo): write your description
        """
        self.model_tester = TFTransfoXLModelTest.TFTransfoXLModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TransfoXLConfig, d_embed=37)

    def test_config(self):
        """
        Test if test test configuration.

        Args:
            self: (todo): write your description
        """
        self.config_tester.run_common_tests()

    def test_transfo_xl_model(self):
        """
        Test if xl_x model.

        Args:
            self: (todo): write your description
        """
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_transfo_xl_model(*config_and_inputs)

    def test_transfo_xl_lm_head(self):
        """
        Test for xlm model was generated.

        Args:
            self: (todo): write your description
        """
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_transfo_xl_lm_head(*config_and_inputs)

    @pytest.mark.slow
    def test_model_from_pretrained(self):
        """
        Test if a model from a pre - built - trained.

        Args:
            self: (todo): write your description
        """
        cache_dir = "/tmp/transformers_test/"
        for model_name in list(TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = TFTransfoXLModel.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
