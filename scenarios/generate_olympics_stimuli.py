"""
Generates olympics stimuli. Optionally takes in a set of existing stimuli that can be used to generate names for the new ones.
Note that this should be run from inside scenarios/
"""
import csv
import os
import random
import numpy as np
import sys
sys.path.append('..') # So we can import RMC
from rmc.constants import *
import rmc.utils as utils

def get_background_parse_sentences(background_parse):
        sentences = [a.split("// ")[-1].strip() for a in background_parse.split("<BEGIN_CODE>")][:-1]
        return "\n".join(sentences)

if __name__ == "__main__":
    base_file = "../scenarios/human-readable/rmc-apr-21-e1-base-stimuli.csv"
    np.random.seed(7)
    random.seed(7)

    # Read all the backgrounds.
    sport_to_backgrounds = {}
    for background_domain in ['tug-of-war', 'canoe-race', 'biathalon']:
        background_domain_file = '../prompts/background-domains/' + background_domain + '_base_skill_to_effort_continuous.txt'
        try:
            background_domain_file = '../prompts/background-domains/' + background_domain + '_base_skill_to_effort_continuous.txt'
            with open(background_domain_file, "r") as f:
                lines = f.readlines()
        except:
            background_domain_file = '../prompts/background-domains/' + background_domain + '_base_skill_to_effort_multimodal_skill_continuous.txt'
            with open(background_domain_file, "r") as f:
                lines = f.readlines()
        background_parse_sentences = get_background_parse_sentences("\n".join(lines))
        sport_to_backgrounds[background_domain] = background_parse_sentences
    
    with open(base_file, "r") as f:
        # Use CSV dict-reader to read the base file.
        reader = csv.DictReader(f)
        stimuli = list(reader)

    sports = ["tug-of-war", "canoe-race", "biathalon"]
    stimuli_map = {
        sport: {}
        for sport in sports
    }
    
    all_stimuli = []
    # Generate all of the stimuli, writing them out to model readable files.
    for base_stim in stimuli:
        base_tag = base_stim["tag"]
        for sport in sports:
            sport_info = sports_map[sport]
            # Get the background.
            rmc_background = sport_to_backgrounds[sport]
            skill_q = sport_info["skill"]
            latent = sport_info["latent"]
            skill_q_txt = skill_variable_map[skill_q]
            latent_q_text = latents[latent]["question"]
            match_token = sport_info["match"]

            tag = f"{sport}_{latent}_{base_tag}_win_how_much_rmc-explicit_continuous_variable"
            # Get a unique hash off of the base_tag
            base_tag_hash = abs(hash(tag)) % (10 ** 8)
            rng = np.random.default_rng(base_tag_hash)
            max_n_athletes = 10
            name_sample = rng.choice(GENDER_NEUTRAL_NAMES, max_n_athletes, replace=False)
            name_map = {}
            for idx, name in enumerate(name_sample):
                if idx <= 8: 
                    name_map[f"P{idx+1}"] = name
                else: 
                    name_map[f"PZ"] = name # todo: if > 10
            
            base_outcomes = base_stim["base-stimulus-outcomes"]
            match_queries = eval(base_stim["match-queries"])
            new_matches = base_stim["newmatch-queries"].split("\n")

            questions = []
            for i, match in enumerate(new_matches):
                comp1, comp2 = match.split(" vs. ")
                comp1 = comp1.replace(", ", " and ")
                comp2 = comp2.replace(", ", " and ")
                comp_q = f"In a new {match_token} later this same day between {comp1} (Team 1) and {comp2} (Team 2), who would win and by how much?"
                questions.append((comp_q, f"new-{i+1}"))
                
                
            parsed_scenario = utils.impute_win_loss(utils.impute_names(base_outcomes, name_map))
            match_txts = [] 
            for match_idx, match_txt in enumerate(parsed_scenario.split('\n')): 
                if match_txt[-1] != ".": match_txt += "."
                match_txt = match_txt.replace(", ", " and ")
                parsed_match_txt = f"In the {MATCH2IDX[f'M{match_idx+1}']} {match_token}, {match_txt}"
                match_txts.append(parsed_match_txt)
            scenario = "\n".join(match_txts)
            
            # implicit bkgrd
            new_entry = {
                    "tag": tag, 
                    "base_tag": base_tag,
                    "base_outcomes": base_outcomes,
                    "questions": [(utils.impute_names(question, name_map), q_tag) for question, q_tag in questions], 
                    "scenario": scenario,
                    "background": f"{rmc_background}",
                    "sport": sport,
                    "skill": skill_q,
                    "latent": latent,
                    "commentary": "",
                    "model_descr_variant": "rmc-explicit"
                }
            all_stimuli.append(new_entry)

# Write all of the stimuli out to model-readable versions.
for stim in all_stimuli:
    question_list = [f"Query {idx+1}: {q}" for idx, (q, q_tag) in enumerate(stim['questions'])]
    question_str = "\n".join(question_list)

    stim_str = f"""BACKGROUND
{stim['background']}

CONDITIONS
{stim['scenario']}

QUERIES
{question_str}
"""

    with open(f"../scenarios/model-readable/olympics/{stim['tag']}.txt", "w") as f:
        f.write(stim_str)