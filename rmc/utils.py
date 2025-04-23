from asyncio import constants
import os
import pathlib
import sys
from urllib import response
# from dotenv import load_dotenv
import numpy as np 
import pandas as pd 
import random
import json
import re
import time
from datetime import datetime
from scipy.stats import pearsonr 
from openai import OpenAI
import os
import seaborn as sns
from matplotlib.pylab import plt
import subprocess
from rmc.constants import *
import tempfile
import itertools
import copy 
import re

from rmc.constants import *

def impute_names(scenario_str, name_set): 
    for person_id, name in name_set.items():
        scenario_str = scenario_str.replace(person_id, name)
    return scenario_str

def impute_win_loss(scenario_str): 
    scenario_str = scenario_str.replace(">", "beat")
    scenario_str = scenario_str.replace("<", "lost to")
    return scenario_str

def read_example(filepth): 
    with open(filepth, "r") as f: 
        return "".join(f.readlines())

def get_scenario_txt(scenario, args): 
    # Current scenario.
    scenario_text = read_example(os.path.join(args.scenario_dir, f"{scenario}.txt"))
    # Append background text if not included
    if "BACKGROUND" not in scenario_text: 
        scenario_bkgrd = args.scenario_bkgrd
        scenario_bkgrd_txt = read_example(os.path.join(args.scenario_bkgrd_dir, f"{scenario_bkgrd}.txt"))
        scenario_text = f"{scenario_bkgrd_txt}\n\n{scenario_text}"
    return scenario_text

def init_experiment_dir(base_dir, scenario, llm_type):
   
    # parse out llm type 
    # note: we may want to change this if we have different llms for different parts of synthesis
    llm_type = llm_type.split("/")[-1] # togther.ai has a backslash for some models
    
    scenario = scenario.split("/")[-1]
    # NOTE -- exp names got too long with prompts ... update to hierarchical/agg level prompt later!!
    experiment_tag = f"{scenario}_{llm_type}"
    full_output_directory = os.path.join(base_dir, experiment_tag)
    pathlib.Path(full_output_directory).mkdir(parents=True, exist_ok=True)
    return experiment_tag, full_output_directory

nth_to_number_map = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10
}

def gold_condition_parse(condition_sentence, matches_to_teams, gold_parses=None):
    if gold_parses is not None:
        if condition_sentence not in gold_parses:
            condition_sentence = condition_sentence + "."
        return gold_parses[condition_sentence]
    else:
        # SPORTS subdomains.
        # In the first race, Ness and Emery beat Blake and Ollie
        # In the fourth race, Ness and Emery lost to Blake and Ollie
        nth_match = re.search("In the (.*?) (.*?),", condition_sentence)
        nth_token, match_token = nth_match.group(1), nth_match.group(2)
        nth_to_number = nth_to_number_map[nth_token]

        beat_lost_token = "beat" if "beat" in condition_sentence else "lost to"

        teams = re.search(f", (.*?) {beat_lost_token} (.*)", condition_sentence)
        team1, team2 = teams.group(1).split("and"), teams.group(2).split("and")
        team1, team2 = [t.lower().strip() for t in team1], [t.lower().strip() for t in team2]
        beat_lost_function = "beat" if "beat" == beat_lost_token else "lost"
        condition_parse = f"condition({beat_lost_function}({{team1: {team1}, team2: {team2}, {match_token}: {nth_to_number}}}))"
        matches_to_teams[nth_to_number] = [team1, team2]
        return condition_parse

def gold_query_parse(query_sentence, sports_domain, query_idx, matches_to_teams):
    match_token = sports_map[sports_domain]["match"]
    # Pretty hard coded. Check the base model to make sure these are right.
    if sports_map[sports_domain]["skill"] in query_sentence:
        athlete = re.search("where do you think (.*) ranks", query_sentence).group(1).lower()
        base_query_string = sports_to_latent_variables_parses[sports_domain][sports_map[sports_domain]["skill"]]
        query_string = f"query{query_idx+1}: " + base_query_string.format(athlete=athlete)
        return query_string
    if latents[sports_map[sports_domain]["latent"]]['token'] in query_sentence:
        athlete = re.search("do you think (.*?)\s+", query_sentence).group(1).strip().lower()
        nth_match = re.search(f"the (.*?)\s+{match_token}", query_sentence).group(1).strip().lower()
        nth_number = nth_to_number_map[nth_match]
        base_query_string = sports_to_latent_variables_parses[sports_domain][sports_map[sports_domain]["latent"]]
        if sports_domain == "diving":
            team = matches_to_teams[nth_number][0] if athlete in matches_to_teams[nth_number][0] else matches_to_teams[nth_number][1]
            teammate = team[0] if athlete == team[1] else team[1]
            query_string = f"query{query_idx+1}: " + base_query_string.format(athlete=athlete, teammate=teammate, match=nth_number)
        else:
            query_string = f"query{query_idx+1}: " + base_query_string.format(athlete=athlete, match=nth_number)
        return query_string
    elif "who would win and by how much" in query_sentence:
        # between Ness and Emery (Team 1) and Quinn and Max (Team 2), who would win and by how much?
        teams = re.search("between (.*) \(Team 1\) and (.*) \(Team 2\)", query_sentence)
        team1, team2 = teams.group(1).split("and"), teams.group(2).split("and")
        team1, team2 = [t.lower().strip() for t in team1], [t.lower().strip() for t in team2]
        base_query_string = sports_to_latent_variables_parses[sports_domain]["new_match"]
        curr_match = len(matches_to_teams) + 1
        matches_to_teams[curr_match] = [team1, team2]
        query_string = f"query{query_idx+1}: " + base_query_string.format(team1=team1, team2=team2, match=curr_match)
        return query_string


def run_webppl(code, tmp_file="temp.wppl", timeout=30, append_utilities=True):
    if append_utilities: 
        starter_code = LIBRARY_FUNCTIONS
    else: starter_code = ""

    webppl_code = starter_code + code + """
var samples = posterior["samples"]
console.dir(samples, {maxArrayLength: null, depth: null})
"""
    # print("running webppl: ",webppl_code )
    
    # print("PATH:", os.environ['PATH'])

    # # Try checking if webppl exists
    # result = subprocess.run(['which', 'webppl'], capture_output=True, text=True)
    # print("webppl location:", result.stdout)

    # tmp_starter = "rand_" + str(np.random.randint(9999))
    # tmp_file_name = tmp_starter + tmp_file
    # # Write the WebPPL code to a temporary file
    # with open(tmp_file_name, "w") as file:
    #     file.write(webppl_code)
        
        # Write the WebPPL code to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wppl") as tmp_file:
        tmp_file.write(webppl_code.encode())
        tmp_file_name = tmp_file.name
    result = subprocess.run(
        ["webppl", tmp_file_name],
        capture_output=True,
        text=True,
        timeout=timeout
    )
    # return result
    
    os.remove(tmp_file_name)
    try:
        if result.stderr != "" or "Error" in result.stdout: 
            print("getting err msg")
            err_msg = "ERROR: " + result.stderr + "\n" + result.stdout
            print("err msg")
            return None, err_msg
        #print("res output: ", result.stdout)
        #keys ={}
        #sample_str = result.stdout
        keys, sample_str = parse_samples(result.stdout, code)
        return keys, sample_str
    except Exception as e: 
        print("error result: ", e)
        err_msg = result
        return None, err_msg 


def parse_samples(result_output, response_code):

    try: 
        # extract out the keys in case of multi-query dict
        # in the case of a single query, keys will hold the function name
        match = find_return_statement(response_code)
        keys = get_keys(match)
        
        # print("sample str: ", result_output)
        
        # print("sample str: ", result_output[:100])
        # print("newline sample str: ", result_output.split(f"[\n"), len(result_output.split(f"[\n")))
        # print("splitting sample str: ", result_output.split(f"[\n")[-1].split("\n]"))
        samples_str = "[" + result_output.split(f"[\n")[-1].split("\n]")[0] + "]"
        
        
        
        
        # come back to for categorical
        #samples_str = "[" + "[\n".join(result_output.split(f"[\n")[1:]).split("\n]")[0] + "]"
        
        
        
        
        
        # print("len: ", len(result_output.split(f"[\n")))
        #return None, "[" + "[\n".join(result_output.split(f"[\n")[1:])
        samples_str = samples_str.replace("\n", "")
        samples_str = samples_str.replace("value", "\"value\"")
        samples_str = samples_str.replace("score", "\"score\"")
        

        # print("sample str pre keys: ", samples_str)
        if keys is not None: 
            for key in keys: 
                samples_str = samples_str.replace(key, f"\"{key}\"")
                
        # print("sample str post keys: ", samples_str)
        # js -> python
        samples_str = samples_str.replace("false", "False")
        samples_str = samples_str.replace("true", "True")
        # print("sample str post T/F: ", samples_str)
        
        samples_str = samples_str.replace("\x1b[32m\'", "'")
        samples_str = samples_str.replace("\'\x1b[39m", "'")
        samples_str = samples_str.replace("\x1b[33m", "")
        samples_str = samples_str.replace("\x1b[39m", "")
        
        return keys, samples_str
    except Exception as e:
        print(e)
        # parsing error 
        return None, None


def find_return_statement(response_code): 
    # Find the last *two* closing parentheses, since the last one is the close of the model and the second to last one should be the end of the return statement. 
    response_code = response_code.split("var posterior")[0]
    pattern = r"(?<![^\n\s])\s*return\s*\{[\s\S]*\}[\s\S]*?\}"
    matches = re.findall(pattern, response_code, re.M)
    if len(matches) != 0:
        # Select the last match and format it
        #print("last matches: ", matches[-2:])
        matches = [match for match in matches if not re.search(r'//.*' + re.escape(match), response_code) and "\nvar posterior" not in match] # if so, it matched a comment in final return
        # Remove the final "}"
        last_match = matches[-1][:-1]
        last_match = last_match.lstrip()[6:].strip()
        if last_match[-1] == ";": last_match = last_match[:-1]# Remove 'return' and the trailing ';'
    else:
        #pattern = r"(?<!\\ )return\s*[\s\S]*?\)" #r"return\s*[\s\S]*?\)"
        # print("response code: ", response_code, response_code.count("\n"))
        # pattern = r"(?<![^\n])\n\s*return\s*[\s\S]*?\)" #r"(?<!\\)(?:\n|^)return\s*[\s\S]*?\)"
        # matches = re.findall(pattern, response_code, re.M)
        pattern = r"return\s*[\s\S]*?\)"#r"(?<![^\n\s])\s*return\s*[\s\S]*?\)"
        matches = re.findall(pattern, response_code, re.M)

        # Filter out matches that are commented out
        
        if len(matches) != 0:
            # be sure the match isn't a "return" in the comment
            # print("MATCHES: ", matches[-2:])
            matches = [match for match in matches if not re.search(r'//.*' + re.escape(match), response_code) and "\nvar posterior" not in match] # if so, it matched a comment in final return
            # print("last matches: ", matches[-2:])
            # Select the last match and format it
            last_match = matches[-1].lstrip()[6:].strip()
            if last_match[-1] == ";": last_match = last_match[:-1]# Remove 'return' and the trailing ';'
        else: 
            print("No match found")
            last_match= None

    # Remove any 'comments'
    last_match = "\n".join([s for s in last_match.split("\n") if not s.strip().startswith("//")])
    return last_match

def get_keys(input_string):
    # Strip leading and trailing whitespace
    input_string = input_string.strip()
    # Handling the dictionary format
    if input_string.startswith("{") and input_string.endswith("}"):
        # Remove new lines and excessive spaces to make the string single-line
        input_string = ' '.join(input_string.split())
        # Extract keys from dictionary format
        keys = re.findall(r"\b(\w+)\s*:", input_string)
    else:
# Handling the non-dictionary format
        # Extract a single key from a non-dictionary format
        key_match = re.findall(r"(\w+)\(", input_string)
        keys = [key_match[0]] if key_match else []

    return keys

def write_checkpoint(particle_idx, particle, parse_metadata, executability, post_samples, err, experiment_dir, args, intermediate_posterior_metadata=None):
    # Write out the model program to be looked at.
    print("Writing checkpoint to: ", f"{experiment_dir}/parse_{particle_idx}.txt")
    with open(f"{experiment_dir}/parse_{particle_idx}.txt", "w") as f:
        f.write(particle.program.to_string())

    parse_metadata["definitions"] = particle.program.definitions
    parse_metadata["conditions"] = particle.program.conditions
    parse_metadata["queries"] = particle.program.queries
 
    # Write out the inference results JSON.
    metadata = {
        "parse": parse_metadata,
        "wm" : {
            "parsed_model" : particle.program.to_string(posterior_samples=args.mean_sampling_budget_per_model, sampling_method=args.sampling_method),
            # TODO: we want to keep commensurability to be able to have multiple sets of posteriors with respect to sets of conditions.
            "official_posterior_samples" : str(post_samples),
            "executability" : executability,
            "err" : err,
            "run_dynamic_posthoc_conditioning": args.run_dynamic_posthoc_conditioning,
            "intermediate_posterior_executabilities": intermediate_posterior_metadata["executabilities"],
            "intermediate_posterior_errs": intermediate_posterior_metadata["errs"],
            "intermediate_posterior_samples" : intermediate_posterior_metadata["posterior_samples"],
        }
    }

    checkpoint_file = os.path.join(experiment_dir, f'inference_results_{particle_idx}.json')
    with open(checkpoint_file, "w") as f:
        json.dump(metadata, f)
    
