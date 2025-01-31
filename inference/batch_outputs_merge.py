import os, json, argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mmlu_small_noaugs_llama_lora')
    parser.add_argument('--batch_outputs_dir', type=str, default='full_precision_results/transformed_mmlupro_reward_results')
    args = parser.parse_args()

    model = args.model
    batch_outputs_dir = os.path.join(args.batch_outputs_dir, 'transformed_mmlupro_with_{}_reward'.format(model))

    files = sorted(os.listdir(batch_outputs_dir))

    for i in range(len(files)):
        assert files[i].split('_')[-1].split('.')[0] == '{}'.format(i)

    data = []
    for file in files:
        with open(os.path.join(batch_outputs_dir, file), 'r', encoding='utf-8') as f:
            data.extend(json.load(f))

    with open(os.path.join(batch_outputs_dir, 'cot_with_{}_rewards.json'.format(model)), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)