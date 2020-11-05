# coding: utf8
def main():
    """
    Main function.

    Args:
    """
    import sys
    if (len(sys.argv) < 4 or len(sys.argv) > 6) or sys.argv[1] not in ["bert", "gpt", "transfo_xl", "gpt2", "xlnet", "xlm"]:
        print(
        "This command line utility let you convert original (author released) model checkpoint to pytorch.\n"
        "It should be used as one of: \n"
        ">> transformers bert TF_CHECKPOINT TF_CONFIG PYTORCH_DUMP_OUTPUT, \n"
        ">> transformers gpt OPENAI_GPT_CHECKPOINT_FOLDER_PATH PYTORCH_DUMP_OUTPUT [OPENAI_GPT_CONFIG], \n"
        ">> transformers transfo_xl TF_CHECKPOINT_OR_DATASET PYTORCH_DUMP_OUTPUT [TF_CONFIG] or \n"
        ">> transformers gpt2 TF_CHECKPOINT PYTORCH_DUMP_OUTPUT [GPT2_CONFIG] or \n"
        ">> transformers xlnet TF_CHECKPOINT TF_CONFIG PYTORCH_DUMP_OUTPUT [FINETUNING_TASK_NAME] or \n"
        ">> transformers xlm XLM_CHECKPOINT_PATH PYTORCH_DUMP_OUTPUT")
    else:
        if sys.argv[1] == "bert":
            try:
                from .convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
            except ImportError:
                print("transformers can only be used from the commandline to convert TensorFlow models in PyTorch, "
                    "In that case, it requires TensorFlow to be installed. Please see "
                    "https://www.tensorflow.org/install/ for installation instructions.")
                raise

            if len(sys.argv) != 5:
                # pylint: disable=line-too-long
                print("Should be used as `transformers bert TF_CHECKPOINT TF_CONFIG PYTORCH_DUMP_OUTPUT`")
            else:
                PYTORCH_DUMP_OUTPUT = sys.argv.pop()
                TF_CONFIG = sys.argv.pop()
                TF_CHECKPOINT = sys.argv.pop()
                convert_tf_checkpoint_to_pytorch(TF_CHECKPOINT, TF_CONFIG, PYTORCH_DUMP_OUTPUT)
        elif sys.argv[1] == "gpt":
            from .convert_openai_original_tf_checkpoint_to_pytorch import convert_openai_checkpoint_to_pytorch
            if len(sys.argv) < 4 or len(sys.argv) > 5:
                # pylint: disable=line-too-long
                print("Should be used as `transformers gpt OPENAI_GPT_CHECKPOINT_FOLDER_PATH PYTORCH_DUMP_OUTPUT [OPENAI_GPT_CONFIG]`")
            else:
                OPENAI_GPT_CHECKPOINT_FOLDER_PATH = sys.argv[2]
                PYTORCH_DUMP_OUTPUT = sys.argv[3]
                if len(sys.argv) == 5:
                    OPENAI_GPT_CONFIG = sys.argv[4]
                else:
                    OPENAI_GPT_CONFIG = ""
                convert_openai_checkpoint_to_pytorch(OPENAI_GPT_CHECKPOINT_FOLDER_PATH,
                                                    OPENAI_GPT_CONFIG,
                                                    PYTORCH_DUMP_OUTPUT)
        elif sys.argv[1] == "transfo_xl":
            try:
                from .convert_transfo_xl_original_tf_checkpoint_to_pytorch import convert_transfo_xl_checkpoint_to_pytorch
            except ImportError:
                print("transformers can only be used from the commandline to convert TensorFlow models in PyTorch, "
                    "In that case, it requires TensorFlow to be installed. Please see "
                    "https://www.tensorflow.org/install/ for installation instructions.")
                raise
            if len(sys.argv) < 4 or len(sys.argv) > 5:
                # pylint: disable=line-too-long
                print("Should be used as `transformers transfo_xl TF_CHECKPOINT/TF_DATASET_FILE PYTORCH_DUMP_OUTPUT [TF_CONFIG]`")
            else:
                if 'ckpt' in sys.argv[2].lower():
                    TF_CHECKPOINT = sys.argv[2]
                    TF_DATASET_FILE = ""
                else:
                    TF_DATASET_FILE = sys.argv[2]
                    TF_CHECKPOINT = ""
                PYTORCH_DUMP_OUTPUT = sys.argv[3]
                if len(sys.argv) == 5:
                    TF_CONFIG = sys.argv[4]
                else:
                    TF_CONFIG = ""
                convert_transfo_xl_checkpoint_to_pytorch(TF_CHECKPOINT, TF_CONFIG, PYTORCH_DUMP_OUTPUT, TF_DATASET_FILE)
        elif sys.argv[1] == "gpt2":
            try:
                from .convert_gpt2_original_tf_checkpoint_to_pytorch import convert_gpt2_checkpoint_to_pytorch
            except ImportError:
                print("transformers can only be used from the commandline to convert TensorFlow models in PyTorch, "
                    "In that case, it requires TensorFlow to be installed. Please see "
                    "https://www.tensorflow.org/install/ for installation instructions.")
                raise

            if len(sys.argv) < 4 or len(sys.argv) > 5:
                # pylint: disable=line-too-long
                print("Should be used as `transformers gpt2 TF_CHECKPOINT PYTORCH_DUMP_OUTPUT [TF_CONFIG]`")
            else:
                TF_CHECKPOINT = sys.argv[2]
                PYTORCH_DUMP_OUTPUT = sys.argv[3]
                if len(sys.argv) == 5:
                    TF_CONFIG = sys.argv[4]
                else:
                    TF_CONFIG = ""
                convert_gpt2_checkpoint_to_pytorch(TF_CHECKPOINT, TF_CONFIG, PYTORCH_DUMP_OUTPUT)
        elif sys.argv[1] == "xlnet":
            try:
                from .convert_xlnet_original_tf_checkpoint_to_pytorch import convert_xlnet_checkpoint_to_pytorch
            except ImportError:
                print("transformers can only be used from the commandline to convert TensorFlow models in PyTorch, "
                    "In that case, it requires TensorFlow to be installed. Please see "
                    "https://www.tensorflow.org/install/ for installation instructions.")
                raise

            if len(sys.argv) < 5 or len(sys.argv) > 6:
                # pylint: disable=line-too-long
                print("Should be used as `transformers xlnet TF_CHECKPOINT TF_CONFIG PYTORCH_DUMP_OUTPUT [FINETUNING_TASK_NAME]`")
            else:
                TF_CHECKPOINT = sys.argv[2]
                TF_CONFIG = sys.argv[3]
                PYTORCH_DUMP_OUTPUT = sys.argv[4]
                if len(sys.argv) == 6:
                    FINETUNING_TASK = sys.argv[5]
                else:
                    FINETUNING_TASK = None

                convert_xlnet_checkpoint_to_pytorch(TF_CHECKPOINT,
                                                    TF_CONFIG,
                                                    PYTORCH_DUMP_OUTPUT,
                                                    FINETUNING_TASK)
        elif sys.argv[1] == "xlm":
            from .convert_xlm_original_pytorch_checkpoint_to_pytorch import convert_xlm_checkpoint_to_pytorch

            if len(sys.argv) != 4:
                # pylint: disable=line-too-long
                print("Should be used as `transformers xlm XLM_CHECKPOINT_PATH PYTORCH_DUMP_OUTPUT`")
            else:
                XLM_CHECKPOINT_PATH = sys.argv[2]
                PYTORCH_DUMP_OUTPUT = sys.argv[3]

                convert_xlm_checkpoint_to_pytorch(XLM_CHECKPOINT_PATH, PYTORCH_DUMP_OUTPUT)

if __name__ == '__main__':
    main()
