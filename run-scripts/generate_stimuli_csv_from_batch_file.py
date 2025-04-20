"""
Convenience script to generate out a human-readable CSV of stimuli from the batch file.
"""

import sys
from pathlib import Path


if __name__ == "__main__": 
    batch_file = "scenarios/batches/jan_22_continuous_batch_explicit.txt"

    with open(batch_file, "r") as f:
        stimuli = f.readlines()
        stimuli = [s.strip() for s in stimuli if len(s.strip()) > 0]
    
    # Sort the stimuli.
    stimuli = sorted(stimuli)
    print(stimuli)

    def get_background_parse_sentences(background_parse):
        sentences = [a.split("// ")[-1].strip() for a in background_parse.split("<BEGIN_CODE>")][:-1]
        return "\n".join(sentences)

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

    import csv 
    with open("scenarios/human-readable/jan_22_continuous_batch_explicit.csv", "w") as f:
        fieldnames = ['full_stimuli_id', 'sport', 'tag', 'original_background', 'rmc_background', 'conditions', 'match-queries', 'newmatch-queries']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for stim in stimuli: 
            sport = stim.split("_")[0]
            # Read in the full stimuli file.
            with open(f"scenarios/model-readable/{stim}.txt", "r") as f:
                lines = f.readlines()
                lines = [l.strip() for l in lines if len(l.strip()) > 0]
                lines_joined = "\n".join(lines)
            original_background = lines_joined.split("BACKGROUND")[-1].split("CONDITIONS")[0].strip()
            conditions = lines_joined.split("CONDITIONS")[1].split("QUERIES")[0].strip()
            queries = lines_joined.split("QUERIES")[1].strip()
            match_queries = queries.split("Query 7: ")[0].strip()
            newmatch_queries = "Query 7: " + queries.split("Query 7: ")[1].strip()
          
            # Extract out the RMC background from the background domains.
            rmc_background = sport_to_backgrounds[sport]
            
            writer.writerow({
                'full_stimuli_id': stim,
                'sport': sport,
                'tag': stim.split("_")[2],
                'original_background': original_background,
                'rmc_background': rmc_background,
                'conditions': conditions,
                'match-queries': match_queries,
                'newmatch-queries': newmatch_queries
            })