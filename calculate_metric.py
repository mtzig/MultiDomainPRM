import json
import os
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt


def calculate_weighted_best_of_n_metrics(json_file_path):
    """
    Calculate Weighted Best-of-N metrics by aggregating RM rewards across identical responses.
    Save metrics and plots.

    Args:
        json_file_path (str): The path to the JSON file.
    
    Returns:
        dict: A dictionary containing the metrics for Weighted Best-of-N.
    """
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Prepare the output directory
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    output_dir = os.path.join(script_dir, "weighted_best_of_n_metrics", os.path.basename(json_file_path).split(".js")[0])
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables for Weighted Best-of-N
    max_samples = 256  # Maximum CoT solutions per problem
    sample_powers = [2 ** i for i in range(9)]  # 2^0 to 2^8
    sample_powers = [a for a in sample_powers if a < len(data[0]['chain_of_thoughts'])] + [len(data[0]['chain_of_thoughts'])]
    aggregation_methods = ['last', 'mean', 'min']
    metrics = {method: {} for method in aggregation_methods}  # Store metrics for each method

    for method in aggregation_methods:
        sampling_results = {n: [] for n in sample_powers}  # Store results for Weighted Best-of-N

        for n in sample_powers:
            if n > max_samples:
                break

            # Repeat sampling 5 times for each size `n`
            for seed in range(10):
                random.seed(seed)  # Set random seed for reproducibility
                correct_count = 0

                # Loop over each question
                for question in data:
                    # Get all CoT solutions and their RM rewards
                    cot_solutions = question['chain_of_thoughts']
                    weighted_scores = {}

                    # Calculate RM reward for each solution based on the aggregation method
                    for cot in cot_solutions:
                        prm_rewards = cot['prm_reward']
                        if method == 'last':
                            rm_reward = prm_rewards[-1]  # Use the last step's prm_reward
                        elif method == 'mean':
                            rm_reward = np.mean(prm_rewards)  # Use the mean of all steps' prm_reward
                        elif method == 'min':
                            rm_reward = np.min(prm_rewards)  # Use the minimum of all steps' prm_reward

                        answer = cot['parsed_answer']
                        if answer not in weighted_scores:
                            weighted_scores[answer] = 0
                        weighted_scores[answer] += rm_reward

                    # Sample N solutions randomly
                    sampled_answers = random.sample(cot_solutions, n)

                    # Aggregate RM rewards for sampled answers
                    sampled_weighted_scores = {}
                    for cot in sampled_answers:
                        prm_rewards = cot['prm_reward']
                        if method == 'last':
                            rm_reward = prm_rewards[-1]
                        elif method == 'mean':
                            rm_reward = np.mean(prm_rewards)
                        elif method == 'min':
                            rm_reward = np.min(prm_rewards)

                        answer = cot['parsed_answer']
                        if answer not in sampled_weighted_scores:
                            sampled_weighted_scores[answer] = 0
                        sampled_weighted_scores[answer] += rm_reward

                    # Select the answer with the highest weighted score
                    best_weighted_answer = max(sampled_weighted_scores.items(), key=lambda x: x[1])[0]

                    # Check correctness of the selected answer
                    for cot in question['chain_of_thoughts']:
                        if cot['parsed_answer'] == best_weighted_answer:
                            if cot['parsed_answer_correctness']:
                                correct_count += 1
                            break

                # Calculate accuracy for this sampling
                accuracy = correct_count / len(data)
                sampling_results[n].append(accuracy)

        # Aggregate results (mean, max, min) for each sampling size
        metrics[method] = {
            n: {
                "mean": np.mean(sampling_results[n]),
                "max": np.max(sampling_results[n]),
                "min": np.min(sampling_results[n]),
                "all": sampling_results[n]
            }
            for n in sampling_results
        }

        # Save results for this method to a JSON file
        metrics_file_path = os.path.join(output_dir, f"metrics_{method}.json")
        with open(metrics_file_path, 'w', encoding='utf-8') as file:
            json.dump(metrics[method], file, indent=4)

        # Plot the results
        x = list(metrics[method].keys())
        y_mean = [metrics[method][n]["mean"] * 100 for n in x]  # Convert to percentages
        y_max = [metrics[method][n]["max"] * 100 for n in x]
        y_min = [metrics[method][n]["min"] * 100 for n in x]

        plt.figure(figsize=(8, 6))
        plt.plot(x, y_mean, '-o', label="Mean Accuracy", color="blue")
        plt.fill_between(x, y_min, y_max, color="blue", alpha=0.2, label="Range (Min-Max)")
        plt.xscale("log", base=2)
        # plt.xticks(x, labels=[f"$2^{{{int(np.log2(n))}}}$" for n in x])
        plt.xticks(x, labels=[f"{n}" for n in x])
        plt.xlabel("Number of sampled CoT solutions")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Weighted Best-of-N Accuracy ({method.capitalize()} RM Reward Aggregation)")
        plt.legend()
        plt.grid(True)

        # Save the plot for this method
        plot_file_path = os.path.join(output_dir, f"accuracy_plot_{method}.png")
        plt.savefig(plot_file_path)
        plt.close()

    return metrics

def calculate_best_of_n_metrics(json_file_path):
    """
    Calculate Best-of-N metrics for choosing the most plausible answer using RM rewards.
    Use three RM reward aggregation methods: last, mean, and min.
    Save metrics and plots for each aggregation method.

    Args:
        json_file_path (str): The path to the JSON file.
    
    Returns:
        dict: A dictionary containing metrics for all RM reward aggregation methods.
    """
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Prepare the output directory
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    output_dir = os.path.join(script_dir, "best_of_n_metrics", os.path.basename(json_file_path).split(".js")[0])
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables for Best-of-N
    max_samples = 256  # Maximum CoT solutions per problem
    sample_powers = [2 ** i for i in range(9)]  # 2^0 to 2^8
    sample_powers = [a for a in sample_powers if a < len(data[0]['chain_of_thoughts'])] + [len(data[0]['chain_of_thoughts'])]
    aggregation_methods = ['last', 'mean', 'min']
    metrics = {method: {} for method in aggregation_methods}  # Store metrics for each method

    for method in aggregation_methods:
        # Store results for this aggregation method
        sampling_results = {n: [] for n in sample_powers}

        # Loop through different values of N (2^0 to 2^8)
        for n in sample_powers:
            if n > max_samples:
                break

            # Repeat sampling 5 times for each size `n`
            for seed in range(10):
                random.seed(seed)  # Set random seed for reproducibility
                correct_count = 0

                # Loop over each question
                for question in data:
                    # Get all CoT solutions and their prm_reward
                    cot_solutions = question['chain_of_thoughts']
                    rewards = []

                    # Calculate RM reward for each solution based on the aggregation method
                    for cot in cot_solutions:
                        prm_rewards = cot['prm_reward']
                        if method == 'last':
                            rm_reward = prm_rewards[-1]  # Use the last step's prm_reward
                        elif method == 'mean':
                            rm_reward = np.mean(prm_rewards)  # Use the mean of all steps' prm_reward
                        elif method == 'min':
                            rm_reward = np.min(prm_rewards)  # Use the minimum of all steps' prm_reward
                        rewards.append((cot['parsed_answer'], rm_reward))

                    # Sample N solutions randomly
                    sampled_rewards = random.sample(rewards, n)

                    # Select the solution with the highest RM reward
                    best_answer = max(sampled_rewards, key=lambda x: x[1])[0]

                    # Check correctness of the selected answer
                    for cot in question['chain_of_thoughts']:
                        if cot['parsed_answer'] == best_answer:
                            if cot['parsed_answer_correctness']:
                                correct_count += 1
                            break

                # Calculate accuracy for this sampling
                accuracy = correct_count / len(data)
                sampling_results[n].append(accuracy)

        # Aggregate results (mean, max, min) for each sampling size
        metrics[method] = {
            n: {
                "mean": np.mean(sampling_results[n]),
                "max": np.max(sampling_results[n]),
                "min": np.min(sampling_results[n]),
                "all": sampling_results[n]
            }
            for n in sampling_results
        }

        # Save results for this method to a JSON file
        metrics_file_path = os.path.join(output_dir, f"metrics_{method}.json")
        with open(metrics_file_path, 'w', encoding='utf-8') as file:
            json.dump(metrics[method], file, indent=4)

        # Plot the results
        x = list(metrics[method].keys())
        y_mean = [metrics[method][n]["mean"] * 100 for n in x]  # Convert to percentages
        y_max = [metrics[method][n]["max"] * 100 for n in x]
        y_min = [metrics[method][n]["min"] * 100 for n in x]

        plt.figure(figsize=(8, 6))
        plt.plot(x, y_mean, '-o', label="Mean Accuracy", color="blue")
        plt.fill_between(x, y_min, y_max, color="blue", alpha=0.2, label="Range (Min-Max)")
        plt.xscale("log", base=2)
        # plt.xticks(x, labels=[f"$2^{{{int(np.log2(n))}}}$" for n in x])
        plt.xticks(x, labels=[f"{n}" for n in x])
        plt.xlabel("Number of sampled CoT solutions")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Best-of-N Accuracy ({method.capitalize()} RM Reward Aggregation)")
        plt.legend()
        plt.grid(True)

        # Save the plot for this method
        plot_file_path = os.path.join(output_dir, f"accuracy_plot_{method}.png")
        plt.savefig(plot_file_path)
        plt.close()

    return metrics



def calculate_majority_voting_metrics_with_sampling(json_file_path):
    """
    Calculate metrics for majority voting accuracy by sampling CoT solutions with sizes 2^0 to 2^8.
    For each sampling size, repeat the sampling 5 times with different random seeds.

    Args:
        json_file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing sampled accuracies (mean, max, min) and overall metrics.
    """
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Prepare the output directory
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    output_dir = os.path.join(script_dir, "majority_voting_metrics", os.path.basename(json_file_path).split(".js")[0])
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables for sampling metrics
    max_samples = 256  # Maximum CoT solutions per problem
    sample_powers = [2 ** i for i in range(9)]  # 2^0 to 2^8
    sample_powers = [a for a in sample_powers if a < len(data[0]['chain_of_thoughts'])] + [len(data[0]['chain_of_thoughts'])]
    sampling_results = {n: [] for n in sample_powers}

    # Outer loop for each sampling size (2^0, 2^1, ..., 2^8)
    for n in sample_powers:
        if n > max_samples:
            break

        # Repeat sampling 5 times for each size `n`
        for seed in range(10):
            random.seed(seed)  # Set random seed for reproducibility
            correct_count = 0

            # Loop over each question
            for question in data:
                # Get all parsed answers and their correctness
                parsed_answers = [cot['parsed_answer'] for cot in question['chain_of_thoughts']]
                correctness_list = [cot['parsed_answer_correctness'] for cot in question['chain_of_thoughts']]

                # Sample `n` solutions randomly
                sampled_indices = random.sample(range(len(parsed_answers)), n)
                sampled_answers = [parsed_answers[i] for i in sampled_indices]
                sampled_correctness = [correctness_list[i] for i in sampled_indices]

                # Perform majority voting on the sampled solutions
                answer_counter = Counter(sampled_answers)
                sampled_majority_answer, _ = answer_counter.most_common(1)[0]
                

                # Check correctness of the sampled majority answer
                sampled_majority_correctness = None
                for i in sampled_indices:
                    if parsed_answers[i] == sampled_majority_answer:
                        sampled_majority_correctness = correctness_list[i]
                        break

                # Update correct count based on majority correctness
                if sampled_majority_correctness:
                    correct_count += 1

            # Calculate accuracy for this sampling
            accuracy = correct_count / len(data)
            sampling_results[n].append(accuracy)

    # Aggregate results (mean, max, min) for each sampling size
    aggregated_results = {
        n: {
            "mean": np.mean(sampling_results[n]),
            "max": np.max(sampling_results[n]),
            "min": np.min(sampling_results[n]),
            "all": sampling_results[n]
        }
        for n in sampling_results
    }

    # Save results to a JSON file
    metrics_file_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_file_path, 'w', encoding='utf-8') as file:
        json.dump(aggregated_results, file, indent=4)

    # Plot the results
    x = list(aggregated_results.keys())
    y_mean = [aggregated_results[n]["mean"] * 100 for n in x]  # Convert to percentages
    y_max = [aggregated_results[n]["max"] * 100 for n in x]
    y_min = [aggregated_results[n]["min"] * 100 for n in x]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y_mean, '-o', label="Mean Accuracy", color="blue")
    plt.fill_between(x, y_min, y_max, color="blue", alpha=0.2, label="Range (Min-Max)")
    plt.xscale("log", base=2)
    # plt.xticks(x, labels=[f"$2^{{{int(np.log2(n))}}}$" for n in x])
    plt.xticks(x, labels=[f"{n}" for n in x])
    plt.xlabel("Number of sampled CoT solutions")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Number of Sampled CoT Solutions")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_file_path = os.path.join(output_dir, "accuracy_plot.png")
    plt.savefig(plot_file_path)
    plt.close()

    return aggregated_results


import os
import json
import matplotlib.pyplot as plt
import numpy as np


def compare_results(file_basename, majority_voting_folder, best_of_n_folder, weighted_best_of_n_folder):
    """
    Compare the results of Majority Voting, Best-of-N, and Weighted Best-of-N
    and plot them on the same graph for each RM reward aggregation method (last, mean, min).
    
    Args:
        majority_voting_folder (str): Folder name of Majority Voting results.
        best_of_n_folder (str): Folder name of Best-of-N results.
        weighted_best_of_n_folder (str): Folder name of Weighted Best-of-N results.
    """
    # Define the output directory
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    output_dir = os.path.join(script_dir, "comparison", file_basename)
    os.makedirs(output_dir, exist_ok=True)

    # Define RM reward aggregation methods
    aggregation_methods = ['last', 'mean', 'min']

    # Define file paths for each method
    majority_voting_path = os.path.join(script_dir, majority_voting_folder, file_basename)
    best_of_n_path = os.path.join(script_dir, best_of_n_folder, file_basename)
    weighted_best_of_n_path = os.path.join(script_dir, weighted_best_of_n_folder, file_basename)

    for method in aggregation_methods:
        # Load metrics for Majority Voting
        majority_metrics_file = os.path.join(majority_voting_path, "metrics.json")
        with open(majority_metrics_file, 'r', encoding='utf-8') as file:
            majority_metrics = json.load(file)

        # Load metrics for Best-of-N
        best_of_n_metrics_file = os.path.join(best_of_n_path, f"metrics_{method}.json")
        with open(best_of_n_metrics_file, 'r', encoding='utf-8') as file:
            best_of_n_metrics = json.load(file)

        # Load metrics for Weighted Best-of-N
        weighted_best_of_n_metrics_file = os.path.join(weighted_best_of_n_path, f"metrics_{method}.json")
        with open(weighted_best_of_n_metrics_file, 'r', encoding='utf-8') as file:
            weighted_best_of_n_metrics = json.load(file)

        # Extract data for plotting
        x = list(map(int, best_of_n_metrics.keys()))  # Sampling sizes (2^0, 2^1, ..., 2^8)
        majority_y = [majority_metrics[str(n)]["mean"] * 100 for n in x]  # Convert to percentages
        best_of_n_y = [best_of_n_metrics[str(n)]["mean"] * 100 for n in x]
        weighted_best_of_n_y = [weighted_best_of_n_metrics[str(n)]["mean"] * 100 for n in x]

        # Plot the results
        plt.figure(figsize=(8, 6))
        plt.plot(x, majority_y, '-o', label="Majority Voting", color="blue")
        plt.plot(x, best_of_n_y, '-o', label="Best-of-N", color="orange")
        plt.plot(x, weighted_best_of_n_y, '-o', label="Weighted Best-of-N", color="green")
        plt.xscale("log", base=2)
        plt.xticks(x, labels=[f"$2^{{{int(np.log2(n))}}}$" for n in x])
        plt.xlabel("Number of sampled CoT solutions (log scale)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Comparison of Voting Methods ({method.capitalize()} RM Reward Aggregation)")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_file_path = os.path.join(output_dir, f"comparison_{method}.png")
        plt.savefig(plot_file_path)
        plt.close()

    print(f"Comparison plots saved to {output_dir}")

if __name__ == "__main__":
    # file_path = "/home/ec2-user/strawberry/full_precision_results/transformed_llama1b_math500_reward_results/transformed_llama1b_math500_with_math_psa_reward/parsed_answer_meta-llama_Llama-3.2-1B-Instruct_HuggingFaceH4_MATH-500_temp0.8_samples256_max_new_tokens_2048_with_math_psa_rewards.json"
    # file_path = "/home/ec2-user/strawberry/full_precision_results/transformed_llama1b_math500_reward_results/transformed_llama1b_math500_with_rlhflow_8b_prm_reward/parsed_answer_meta-llama_Llama-3.2-1B-Instruct_HuggingFaceH4_MATH-500_temp0.8_samples256_max_new_tokens_2048_with_rlhflow_8b_prm_rewards.json"
    file_path = "full_precision_results/transformed_prm800k_small_test_set_reward_results/transformed_prm800k_small_test_set_with_rlhflow-ds_qwen_fulltune_reward/prm800_best_of_n_100_with_rlhflow-ds_qwen_fulltune_rewards.json"
    majority_voting_metrics = calculate_majority_voting_metrics_with_sampling(file_path)
    best_of_n_metrics = calculate_best_of_n_metrics(file_path)
    weighted_best_of_n_metrics = calculate_weighted_best_of_n_metrics(file_path)

    compare_results(file_basename = os.path.basename(file_path).split(".js")[0],
                    majority_voting_folder="majority_voting_metrics",
                    best_of_n_folder="best_of_n_metrics",
                    weighted_best_of_n_folder="weighted_best_of_n_metrics"
                    )
