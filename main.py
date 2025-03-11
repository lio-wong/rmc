"""
main.py | Entrypoint to the RMC pipeline. 

Developer notes: designed for parallelism to https://github.com/collinskatie/marshaling 
""" 

import argparse
import json
import random
import numpy as np

import rmc.synthesize_ppl as synthesize_ppl
import rmc.utils as utils

parser = argparse.ArgumentParser()

#### Experiment core input parameters
parser.add_argument('--scenario', type=str, help='Input problem scenario.')
parser.add_argument('--background_domains', type=str, nargs="+", default=['tug_of_war', 'jump'], help='Background knowledge domains to seed few shot prompt.')

# World model evaluation parameters.
parser.add_argument('--mean_sampling_budget_per_model', type=int, default=5000, help='Mean number of samples per model to evaluate.')
parser.add_argument('--sampling_method', type=str, default="rejection")



def answer_questions(scenario, experiment_dir, args):
    # TODO: alternative models using LLMs can go here.
    return rmc(scenario, experiment_dir, args)
 
def rmc(scenario, experiment_dir, args):
    # Synthesize programs from scenario text.
    programs = synthesize_ppl.parse(scenario, experiment_dir, args)
    # Inferences to answer questions given world models.
   
if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(args.random_seed)
    rng = np.random.default_rng(args.random_seed)

    experiment_tag, experiment_dir = utils.init_experiment_dir(
        base_dir=args.base_dir,
        scenario=args.scenario,
        background_domains=args.background_domains, 
        background_domain_prompt_type=args.background_domain_prompt_type,
        llm_type=args.llm)
    
    # save the params to that folder
    with open(f"{experiment_dir}/params.json", "w") as f: json.dump(vars(args), f)

    synthesis_metadata = answer_questions(
                scenario = args.scenario,
                experiment_dir=experiment_dir,
                args = args,
            )



