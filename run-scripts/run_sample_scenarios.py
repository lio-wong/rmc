"""
Run experiments over many scenarios, sampling scenarios from the rest of the dataset to use as prompts. Controls sampling of the scenarios.
"""
import argparse
from collections import defaultdict
import random
import subprocess 
parser = argparse.ArgumentParser()

parser.add_argument("--run_file", type=str, help="File with commands to run.")
parser.add_argument('--scenario_file', type=str, default='scenarios/batches/base_scenarios.txt')
parser.add_argument('--scenario_dir', type=str, default='scenarios/model-readable')
parser.add_argument('--llms', nargs='+', type=str, 
                    default=["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"],
                    help='List of LLM models to use')
parser.add_argument('--random_seed', type=int, default=7,
    help='Random seed')
parser.add_argument('--debug_run_first_k', type=int, default=-1, 
                    help='By default (-1), run all scenarios. Otherwise, just run the first k.')
parser.add_argument('--mean_sampling_budget_per_model', type=int, default=5000, help='Mean number of samples per model to evaluate.')
parser.add_argument('--sampling_method', type=str, default="rejection")
parser.add_argument("--base_dir", type=str, default="rmc-experiments/", help='Base output directory for runs.')

if __name__ == "__main__": 
    # Handwritten background parses.
    background_parses = [
        "tug-of-war_base_skill_to_effort_continuous",
        "canoe-race_base_skill_to_effort_continuous",
        "biathalon_base_skill_to_effort_multimodal_skill_continuous",
    ]

    args = parser.parse_args()
    random.seed(args.random_seed)
    
    with open(args.scenario_file, "r") as f:
        scenarios = f.readlines()
        # Backward compatability with older scenario files.
        scenarios = [f"{stim.split('.txt')[0]}".strip() for stim in scenarios]
        scenarios = [s for s in scenarios if len(s) > 0]
    
    llms = args.llms

    scenarios_to_run = scenarios
    if args.debug_run_first_k != -1: 
        scenarios_to_run = scenarios[:args.debug_run_first_k]
    print("SCENARIOS: ", scenarios, len(scenarios))

    for scenario in scenarios_to_run: 
        print("RUNNING FOR SCENARIO: ", scenario)

        # Randomly select scenarios from other domains from the rest of the scenario file.
        scenario_domain = scenario.split("_")[0]
        domains_to_scenarios = defaultdict(list)
        for other_scenario in scenarios: 
            other_scenario_domain = other_scenario.split("_")[0]
            if other_scenario_domain != scenario_domain: 
                domains_to_scenarios[other_scenario_domain].append(other_scenario)
        scenario_prompts = []
        for d in domains_to_scenarios:
            if d != scenario_domain:
                scenario_prompts += list(random.sample(domains_to_scenarios[d], 1))

        for llm in llms:
            with open(args.run_file, "r") as f:
                commands = f.readlines()
            for line in commands:
                    line = line.strip()
                    line = line.replace("$MEAN_SAMPLING_BUDGET", str(args.mean_sampling_budget_per_model))
                    line = line.replace("$SCENARIO_DIR", f"{args.scenario_dir}")
                    line = line.replace("$SCENARIO", f"{scenario}")
                    line = line.replace("$LLM", f"{llm}")
                    
                    line = line.replace("$SAMPLING_METHOD", f"{args.sampling_method}")
                    line = line.replace("$BASE_DIR", f"{args.base_dir}")
                    line = line.replace('$BACKGROUND_PARSES', " ".join(background_parses))
                    line = line.replace("$PROMPTS", " ".join(scenario_prompts))
                    
                    print("\n\nRunning ...", line)
                    command_args = line.split()
                    print("\n\nArgs ... ", command_args)
                    subprocess.run(command_args)






    
