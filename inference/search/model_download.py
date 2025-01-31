import argparse
from transformers import BitsAndBytesConfig

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="download PRM model checkpoints")

    # Add argument for the directory paths and test_prm
    parser.add_argument('--test_prm', type=str, choices=['llemma', 'math_shepherd', "gpt4o_real", "reasoneval_7b", "math_psa", "skywork_o1_open_prm_qwen2.5_7b", "rlhflow_8b_prm", "rlhflow_deepseek", "prm800k_qwen_alt_lora", "prm800k_llama_lora", "prm800k_llama_fulltune", "prm800k_qwen_fulltune", "rlhflow-ds_qwen_fulltune", "v4_llama_lora", "v4_qwen_lora", "sciqqa_noaugs_qwen_lora", "sciqqa_noaugs_llama_lora", "v5_qwen_lora", "v5_llama_lora", "v6_llama_lora", "sciqqa_noaugs_masked_qwen_lora", "v7_noaugs_qwen_lora", "sciqqa_augs_llama_lora", "mmlu_noaugs_llama_lora", "v7_noaugs_llama_lora", "mmlu_onlyaugs_llama_lora", "v7_onlyaugs_llama_lora", "sciqqa_onlyaugs_llama_lora", "mmlu_augs_llama_lora", "mmlu_math_noaugs_llama_lora", "mmlu_small_noaugs_llama_lora", "mmlu_heo_augs_llama_lora", "mmlu_noaugs_llamabase_lora", "mmlu_noaugs_qwen_lora"], required=True, help="PRM model to use")
    parser.add_argument("--four_bit", action="store_true")
    args = parser.parse_args()

    test_prm = args.test_prm

    print('\nModel {} loading test. Download it if not yet.'.format(test_prm))

    if args.four_bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        quantization_config = None

    if test_prm == "llemma":
        from prm_models.llemma_7b_prm import Llemma7bPRM
        prm = Llemma7bPRM(
            aggregation="full", 
            quantization_config=quantization_config
        )
    elif test_prm == "math_shepherd":
        from prm_models.mistral_7b_prm import Mistral7bPRM
        prm = Mistral7bPRM(
            aggregation="full", 
            quantization_config=quantization_config
        )
    elif test_prm == "gpt4o_real":
        from prm_models.gpt4o_prm_real import gpt4o_prm
        prm = gpt4o_prm(
                aggregation="full", 
        )
    elif test_prm == "reasoneval_7b":
        from prm_models.reasoneval_7b import ReasonEval7bPRM
        prm = ReasonEval7bPRM(
            aggregation="full", 
            quantization_config=quantization_config
        )
    elif test_prm == "math_psa":
        from prm_models.math_psa import math_psa_prm
        prm = math_psa_prm(
            aggregation="full", 
            quantization_config=quantization_config
        )
    elif test_prm == "skywork_o1_open_prm_qwen2.5_7b":
        from prm_models.skywork_o1_open_prm_qwen25_7b import Qwen7bPRM
        prm = Qwen7bPRM(
            aggregation="full", 
            quantization_config=quantization_config
        )
    elif test_prm == "rlhflow_8b_prm":
        from prm_models.rlhflow_8B_prm import RLHflow8bPRM
        prm = RLHflow8bPRM(
            aggregation="full", 
            quantization_config=quantization_config
        )
    elif test_prm == "rlhflow_deepseek":
        from prm_models.deepseek_8B_prm import Deepseek_RLHflow8bPRM
        prm = Deepseek_RLHflow8bPRM(
            aggregation="full", 
            quantization_config=quantization_config
        )
    elif test_prm in ["prm800k_llama_lora"]:
        from prm_models.prm_llama import test_prm_dual
        prm = test_prm_dual(
            aggregation="full", 
            model_id="mtzig/{}".format(test_prm)
        )
    elif test_prm in ["prm800k_qwen_alt_lora"]:
        from prm_models.prm_qwen import test_prm_dual
        prm = test_prm_dual(
            aggregation="full", 
            model_id="mtzig/{}".format(test_prm)
        )
    elif test_prm in ["prm800k_qwen_fulltune", "rlhflow-ds_qwen_fulltune", "v4_qwen_lora", "v5_qwen_lora", "sciqqa_noaugs_qwen_lora", "sciqqa_noaugs_masked_qwen_lora", "v7_noaugs_qwen_lora", "mmlu_noaugs_qwen_lora"]:
        from prm_models.prm_qwen import test_prm_dual
        prm = test_prm_dual(
            aggregation="full", 
            model_id="Daewon0808/{}".format(test_prm)
        )
    elif test_prm in ["prm800k_llama_fulltune", "sciqqa_noaugs_llama_lora", "v4_llama_lora", "v5_llama_lora", "v6_llama_lora", "sciqqa_augs_llama_lora", "mmlu_noaugs_llama_lora", "v7_noaugs_llama_lora", "mmlu_onlyaugs_llama_lora", "v7_onlyaugs_llama_lora", "sciqqa_onlyaugs_llama_lora", "mmlu_augs_llama_lora", "mmlu_math_noaugs_llama_lora", "mmlu_small_noaugs_llama_lora", "mmlu_heo_augs_llama_lora", "mmlu_noaugs_llamabase_lora"]:
        from prm_models.prm_llama import test_prm_dual
        prm = test_prm_dual(
            aggregation="full", 
            model_id="Daewon0808/{}".format(test_prm)
        ) 
    
    print('Model {} loading test successful!\n'.format(test_prm))