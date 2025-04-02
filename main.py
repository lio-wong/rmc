"""
main.py | Entrypoint to the RMC pipeline. 

Developer notes: designed for parallelism to https://github.com/collinskatie/marshaling 
""" 

import argparse
import json
import random
import numpy as np

import rmc.synthesize_ppl as synthesize_ppl
import rmc.inference_ppl as inference_ppl
import rmc.utils as utils

parser = argparse.ArgumentParser()

#### Experiment core input parameters
parser.add_argument('--scenario', type=str, help='Input problem scenario.')
parser.add_argument('--background_parses', type=str, nargs="+", default=['tug_of_war', 'jump'], help='Background knowledge domains to seed few shot prompt.')
parser.add_argument('--background_domains', type=str, nargs="+", default=['tug_of_war', 'jump'], help='Background knowledge domains to seed few shot prompt.')

### World model low-level parameters
parser.add_argument('--random_seed', type=int, default=7,
    help='Random seed')
parser.add_argument('--llm', type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    help='LLM to run')

# World model evaluation parameters.
parser.add_argument('--mean_sampling_budget_per_model', type=int, default=5000, help='Mean number of samples per model to evaluate.')
parser.add_argument('--sampling_method', type=str, default="rejection")

#### Experiment utilities
parser.add_argument('--background_domains_dir', type=str, default="prompts/background-domains")
parser.add_argument('--scenario_dir', type=str, default="scenarios/model-readable")
parser.add_argument('--base_dir', type=str, default="rmc-experiments/",
    help='Base output directory for runs.')

parser.add_argument("--replace_background_with_background_parses", action="store_true", help="If true, replace background knowledge with the version in the background parses.")


def answer_questions(scenario, experiment_dir, rng, args):
    # TODO: alternative models using LLMs can go here.
    return rmc(scenario, experiment_dir, rng, args)
 
def rmc(scenario, experiment_dir, rng, args):
    # Synthesize programs from scenario text.
    particles, parse_metadata = synthesize_ppl.parse(scenario, background_domains=args.background_domains, experiment_dir=experiment_dir, rng=rng, args=args)

    # Inferences to answer questions given world models.
    # TODO: consider parallelizing.
    for particle_idx, particle in enumerate(particles):
        executability, post_samples, err = inference_ppl.evaluate_probabilistic_program(program=particle.program, sampling_budget=args.mean_sampling_budget_per_model, sampling_method=args.sampling_method)

        utils.write_checkpoint(particle_idx, particle, parse_metadata, executability, post_samples, err, experiment_dir, args)
   
if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(args.random_seed)
    rng = np.random.default_rng(args.random_seed)

    experiment_tag, experiment_dir = utils.init_experiment_dir(
        base_dir=args.base_dir,
        scenario=args.scenario,
        llm_type=args.llm)
    
    # save the params to that folder
    with open(f"{experiment_dir}/params.json", "w") as f: json.dump(vars(args), f)

    synthesis_metadata = answer_questions(
                scenario = args.scenario,
                experiment_dir=experiment_dir,
                rng=rng,
                args = args,
            )



