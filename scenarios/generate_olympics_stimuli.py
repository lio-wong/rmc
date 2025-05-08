"""
Generates olympics stimuli. Optionally takes in a set of existing stimuli that can be used to generate names for the new ones.
Note that this should be run from inside scenarios/
"""
import csv
import json
import os
import random
import numpy as np
import sys
sys.path.append('..') # So we can import RMC
from rmc.constants import *
import rmc.utils as utils

def get_background_parse_sentences(background_parse):
        sentences = [a.split("// ")[-1].strip() for a in background_parse.split("<BEGIN_CODE>")][:-1]
        return sentences

if __name__ == "__main__":
    base_file = "../scenarios/human-readable/rmc-apr-22-e1-base-stimuli.csv"
    np.random.seed(7)
    random.seed(7)

    SPORTS = ['tug-of-war', 'canoe-race', 'biathalon']

    ## Read in all of the background sentences.
    sport_to_backgrounds = {}
    for background_domain in SPORTS:
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

    # STIMULI_BATCH_MAP
    stimuli_map = {
        sport: {

        }
        for sport in SPORTS
    }
    all_stimuli_flat = [] # Flat list of all of the stimuli.

    # Generate all of the stimuli, writing them out to model readable files.
    for base_stim_idx, base_stim in enumerate(stimuli):
        base_tag = base_stim["tag"]
        for sport in SPORTS:
            sport_info = sports_map[sport]
            # Get the background.
            rmc_background = sport_to_backgrounds[sport]
            skill_q = sport_info["skill"]
            latent = sport_info["latent"]
            skill_q_txt = skill_variable_map[skill_q]
            latent_q_txt = latents[latent]["question"]
            latent_scale = latents[latent]["scale"]
            match_token = sport_info["match"]

            base_outcomes = base_stim["base-stimulus-outcomes"]
            match_queries = eval(base_stim["match-queries"])
            new_matches = base_stim["newmatch-queries"].split("\n")

            for player_idx, player_of_interest in enumerate(list(match_queries.keys())):
                tag = f"{sport}_{latent}_{base_tag}_{player_of_interest}_win_how_much_rmc-explicit_continuous_variable"
                # Get a unique hash off of the base_tag so that they're comparable across sports.
                base_tag_hash = abs(hash(base_tag)) % (10 ** 8)
                rng = np.random.default_rng(base_tag_hash)
                max_n_athletes = 10
                name_sample = rng.choice(GENDER_NEUTRAL_NAMES, max_n_athletes, replace=False)
                name_map = {}
                for idx, name in enumerate(name_sample):
                    if idx <= 8: 
                        name_map[f"P{idx+1}"] = name
                    else: 
                        name_map[f"PZ"] = name # todo: if > 10
            
                questions = []
                player_of_interest_name = name_map[player_of_interest]

                # Skill question for player of interest.
                questions.extend([(SKILL_QUESTION.format(player=player_of_interest, skill_q_txt=skill_q_txt, back_tag=BACKTAG_BOLD), f"skill-{player_idx+1}", SKILL_QUESTION_SCALE)]) 

                # Latent question for player of interest.
                match_of_interest = match_queries[player_of_interest]
                questions.append((latent_q_txt.format(player=player_of_interest, match_idx=MATCH2IDX[match_of_interest], match_token=match_token, back_tag=BACKTAG_BOLD), f"latent-{player_idx+1}", latent_scale))
            
                # Get the new match of interesting.
                new_match_of_interest = new_matches[player_idx]
                comp1, comp2 = new_match_of_interest.split(" vs. ")
                comp1 = comp1.replace(", ", " and ")
                comp2 = comp2.replace(", ", " and ")
                questions.append((NEW_MATCH_LIKELIHOOD_QUESTION.format(match_token=match_token, comp1=comp1, comp2=comp2, back_tag=BACKTAG_BOLD), f"newmatch-{player_idx+1}", NEW_MATCH_LIKELIHOOD_QUESTION_SCALE))  

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
                        "questions": [(utils.impute_names(question, name_map), q_tag, q_scale) for question, q_tag, q_scale in questions], 
                        "scenario": scenario.split("\n"),
                        "background": rmc_background,
                        "sport": sport,
                        "skill": skill_q,
                        "latent": latent,
                        "commentary": "",
                        "model_descr_variant": "rmc-explicit",
                        "player_idx" : player_idx,
                        "player_of_interest": player_of_interest,
                        "base_stim_idx": base_stim_idx,
                        "player_of_interest_name": player_of_interest_name
                    }
                
                all_stimuli_flat.append(new_entry)
                if player_idx not in stimuli_map[sport]:
                    stimuli_map[sport][player_idx] = []
                stimuli_map[sport][player_idx].append(new_entry)


# Write all of the stimuli out to model-readable versions.
for stim in all_stimuli_flat:
    question_list = [f"Query {idx+1}: {q}" for idx, (q, q_tag, _) in enumerate(stim['questions'])]
    question_str = "\n".join(question_list)
    background_str = "\n".join(stim['background'])
    conditions_str = "\n".join(stim['scenario'])

    stim_str = f"""BACKGROUND
{background_str}

CONDITIONS
{conditions_str}

QUERIES
{question_str}"""

    with open(f"../scenarios/model-readable/olympics/{stim['tag']}.txt", "w") as f:
        f.write(stim_str)


# Just dump this out to a batch file. We will use odds and evens as conditions.
with open("../scenarios/human-experiment-batches/apr-22_rmc-apr-22-e1-base-stimuli.json", "w") as f:
    json.dump(stimuli_map, f)

# Make a list of all of the stimuli.
with open("../scenarios/batches/apr-22_rmc-apr-22-e1-base-stimuli.txt", "w") as f:
    for stim in all_stimuli_flat:
        f.write(f"{stim['tag']}\n")