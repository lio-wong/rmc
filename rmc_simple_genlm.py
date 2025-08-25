"""
Simple implementation of RMC that assumes that the 'translation transcript' is a sequentially growing set of alternating (τ, π) tuples.
Inputs: JSON stimulus file with a list of stimuli containing their background domains and observations.
Outputs: Translations of the stimuli into WebPPL code and inferences over the questions.
"""

import argparse
import json
import random
import numpy as np

import rmc.utils as utils

parser = argparse.ArgumentParser()

#### Experiment utilities
parser.add_argument('--base_dir', type=str, default="rmc-experiments",
    help='Base output directory for runs.')
parser.add_argument('--base_experiment_tag', type=str, default="demo",
    help='Base sub-dir for runs.')
parser.add_argument('--random_seed', type=int, default=0,
    help='Random seed')

### LLM parameters
parser.add_argument('--llm', type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    help='LLM to run')


if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(args.random_seed)
    rng = np.random.default_rng(args.random_seed)

    experiment_tag, experiment_dir = utils.init_experiment_dir(
        base_dir=args.base_dir,
        base_experiment_tag=args.base_experiment_tag,
        llm_type=args.llm)
    # save the params to that folder
    with open(f"{experiment_dir}/params.json", "w") as f: json.dump(vars(args), f)

    