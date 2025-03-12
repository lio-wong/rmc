import datetime
import os
import pathlib
import re

from rmc.constants import *

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

def gold_condition_parse(condition_sentence, matches_to_teams):
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
        print(athlete, query_sentence)
        base_query_string = sports_to_latent_variables_parses[sports_domain][sports_map[sports_domain]["skill"]]
        query_string = f"query{query_idx+1}: " + base_query_string.format(athlete=athlete)
        return query_string
    if latents[sports_map[sports_domain]["latent"]]['token'] in query_sentence:
        print(query_sentence)
        athlete = re.search("do you think (.*?)\s+", query_sentence).group(1).strip().lower()
        nth_match = re.search(f"the (.*?)\s+{match_token}", query_sentence).group(1).strip().lower()
        print(athlete, nth_match)
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