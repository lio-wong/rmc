import datetime
import os
import pathlib


def init_experiment_dir(base_dir, scenario, background_domains, background_domain_prompt_type, 
                        llm_type):
   
    # parse out llm type 
    # note: we may want to change this if we have different llms for different parts of synthesis
    llm_type = llm_type.split("/")[-1] # togther.ai has a backslash for some models
    
    scenario = scenario.split("/")[-1]
    # NOTE -- exp names got too long with prompts ... update to hierarchical/agg level prompt later!!
    experiment_tag = f"{scenario}_{llm_type}_{background_domain_prompt_type}"
    full_output_directory = os.path.join(base_dir, experiment_tag)
    pathlib.Path(full_output_directory).mkdir(parents=True, exist_ok=True)
    return experiment_tag, full_output_directory