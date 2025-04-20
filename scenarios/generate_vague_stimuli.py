"""
Generates stimuli with vague adjectives from base stimuli.
"""
POSITIVE_VAGUE_ADJECTIVES = [
    "strong",
    "not weak",
    "somewhat strong",
    "pretty strong",
    "very strong",
    "not so weak",
    "not that weak",
    "not very weak",
]
NEGATIVE_VAGUE_ADJECTIVES = [
    "weak",
    "not strong",
    "somewhat weak",
    "pretty weak",
    "very weak",
    "not so strong",
    "not that strong",
    "not very strong",
]

import sys
from pathlib import Path
import csv 

def extract_players_from_conditions(conditions, base_stimulus_outcomes_string):
    base_outcomes = base_stimulus_outcomes_string.split("\n")
    # Replace the 'sign' because we just want the list of players.
    base_outcomes = [o.replace("<", ">") for o in base_outcomes]
    # Splti them on the sign.
    base_outcomes = [o.split(">") for o in base_outcomes]
    # Further split them on the comma.
    base_outcomes = [[each_outcome.split(",") for each_outcome in o] for o in base_outcomes]

    # Now get them out of the condition sentences.
    conditions = conditions.split("\n")
    # Remove the part after the comma.
    conditions = [c.split(",")[-1].strip() for c in conditions]
    # Replace 'lost to' with 'beat'.
    conditions = [c.replace("lost to", "beat").replace(".", "") for c in conditions]
    # Split them on the 'beat'.
    conditions = [c.split("beat") for c in conditions]

    # Further split them on the and.
    conditions = [[each_condition.split("and") for each_condition in c] for c in conditions]

    player_to_name_map = {}
    for (o, c) in zip(base_outcomes, conditions):
        for pair in list(zip(o, c)):
            for (p_num, p_name) in zip(*pair):
                player_to_name_map[p_num.strip()] = p_name.strip()
    return player_to_name_map


def get_background_parse_sentences(background_parse):
        sentences = [a.split("// ")[-1].strip() for a in background_parse.split("<BEGIN_CODE>")][:-1]
        return "\n".join(sentences)

if __name__ == "__main__":
    base_file = "scenarios/human-readable/rmc-model-olympics-stimuli - rmc-vague-adjectives-e1-vague-all-base.csv"

     # Read all the backgrounds.
    sport_to_backgrounds = {}
    for background_domain in ['tug-of-war', 'canoe-race', 'biathalon']:
        background_domain_file = 'prompts/background-domains/' + background_domain + '_base_skill_to_effort_continuous.txt'
        try:
            background_domain_file = 'prompts/background-domains/' + background_domain + '_base_skill_to_effort_continuous.txt'
            with open(background_domain_file, "r") as f:
                lines = f.readlines()
        except:
            background_domain_file = 'prompts/background-domains/' + background_domain + '_base_skill_to_effort_multimodal_skill_continuous.txt'
            with open(background_domain_file, "r") as f:
                lines = f.readlines()
        background_parse_sentences = get_background_parse_sentences("\n".join(lines))
        sport_to_backgrounds[background_domain] = background_parse_sentences

    with open(base_file, "r") as f:
        # Use CSV dict-reader to read the base file.
        reader = csv.DictReader(f)
        stimuli = list(reader)
    
    with open("scenarios/human-readable/rmc-vague-adjectives-e1-vague-all.csv", "w") as f:
        fieldnames = ['full_stimuli_id', 'sport', 'tag', 'rmc_background', 'vague_sentence', 'conditions', 'match-queries', 'newmatch-queries']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for base_stim in stimuli:
            tag = base_stim["tag"]
            base_stim_file = "scenarios/model-readable/" + f"tug-of-war_effort_{tag}_win_how_much_explicit_continuous_variable" + ".txt"

            # Read in the base stimuli file.
            with open(base_stim_file, "r") as f:
                lines = f.readlines()
                lines = [l.strip() for l in lines if len(l.strip()) > 0]
                lines_joined = "\n".join(lines)

            original_background = lines_joined.split("BACKGROUND")[1].split("CONDITIONS")[0].strip()
            conditions = lines_joined.split("CONDITIONS")[1].split("QUERIES")[0].strip()
            queries = lines_joined.split("QUERIES")[1].strip()
            match_queries = queries.split("Query 7: ")[0].strip()
            newmatch_queries = "Query 7: " + queries.split("Query 7: ")[1].strip()

            player_to_name_map = extract_players_from_conditions(conditions, base_stim['base-stimulus-outcomes'])

            # Extract out the RMC background from the background domains.
            rmc_background = sport_to_backgrounds['tug-of-war']
            
            # This is an extremely hacky way to get the players out  and should be generalized.
            for player in base_stim['vague-adjective-players-of-interest'].split(","):
                player_name = player_to_name_map[player.strip()]

                for adj in POSITIVE_VAGUE_ADJECTIVES + NEGATIVE_VAGUE_ADJECTIVES:
                    new_sentence = f"{player_name} is {adj}."

                    new_full_stim_id = f"tug-of-war_effort_{tag}_win_how_much_explicit_continuous_variable_{player_name}_{adj}"
                    writer.writerow({
                        'full_stimuli_id': new_full_stim_id,
                        "sport": "tug-of-war",
                        "tag": tag,
                        "rmc_background": original_background,
                        "vague_sentence": new_sentence,
                        "conditions": conditions,
                        "match-queries": match_queries,
                        "newmatch-queries": newmatch_queries
                    })

                    # Write out the new stimulus file.
                    new_stim_file = "scenarios/model-readable/" + f"tug-of-war_effort_{tag}_win_how_much_explicit_continuous_variable_{player_name}_{adj}" + ".txt"
                    with open(new_stim_file, "w") as f:
                        f.write(f"BACKGROUND\n{rmc_background}\n\nCONDITIONS\n{conditions}\n{new_sentence}\n\nQUERIES\n{queries}")
        
        

    


