# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert XXX checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch

from transformers import XxxConfig, XxxForPreTraining, load_tf_weights_in_xxx

import logging
logging.basicConfig(level=logging.INFO)

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, xxx_config_file, pytorch_dump_path):
    """
    Converts checkpoint_tf_checkpoint_path.

    Args:
        tf_checkpoint_path: (str): write your description
        xxx_config_file: (str): write your description
        pytorch_dump_path: (str): write your description
    """
    # Initialise PyTorch model
    config = XxxConfig.from_json_file(xxx_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = XxxForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_xxx(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tf_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the TensorFlow checkpoint path.")
    parser.add_argument("--xxx_config_file",
                        default = None,
                        type = str,
                        required = True,
                        help = "The config json file corresponding to the pre-trained XXX model. \n"
                            "This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path,
                                     args.xxx_config_file,
                                     args.pytorch_dump_path)
