import os
import sys

from rmc import constants, utils
sys.path.append("../hfppl")

def get_scenario_conditions(scenario):
    condition_sentences = scenario.split('CONDITIONS')[1].split('QUERIES')[0].split("\n")
    condition_sentences = [s.strip() for s in condition_sentences if len(s.strip()) > 0]
    # remove ending periods if there (b/c not always generated + leading to parse issues)
    condition_sentences = [s[:-1] if s[-1] == "." else s for s in condition_sentences]
    return condition_sentences

def get_scenario_queries(scenario):
    query_sentences = scenario.split('QUERIES')[1].split("\n")
    query_sentences = [s.strip() for s in query_sentences if len(s.strip()) > 0 and not s.startswith("//")]
    return query_sentences

def construct_background_domains_prompt(scenario, rng, background_domains, args):
    # Add in a header.

    # Add in library functions.

    shuffled_background_domains = rng.permutation(background_domains)
    return "\n".join(construct_background_domain_prompt(background_domain, args) for background_domain in shuffled_background_domains)

def construct_background_domain_prompt(background_domain, args):
    sports_domain = background_domain.split("_")[0]
    background_scenario_text = utils.get_scenario_txt(background_domain, args)

    p = constants.START_PARSE_TOKEN + "\n"
    # Construct the background parse.
    background_parse_file = [b for b in args.background_parses if sports_domain in b][0]
    with open(os.path.join(args.background_domains_dir, background_parse_file + ".txt"), "r") as f:
        background_parse = f.read().strip()
    p += background_parse + "\n\n"

    # Construct the conditions parse.
    condition_sentences = get_scenario_conditions(background_scenario_text)
    # Construct the queries parse.
    query_sentences = get_scenario_queries(background_scenario_text)
    matches_to_teams = {}
    condition_parses = [utils.gold_condition_parse(c, matches_to_teams) for c in condition_sentences]
    query_parses = [utils.gold_query_parse(query_sentence=q, sports_domain=sports_domain, query_idx=idx, matches_to_teams=matches_to_teams) 
                       for idx, q in enumerate(query_sentences)]
    
    condition_parse = "\n".join([f"// {s}\n{constants.START_SINGLE_PARSE_TOKEN}\n{p}\n{constants.END_SINGLE_PARSE_TOKEN}\n" for (s, p) in zip(condition_sentences, condition_parses)])

    query_parse = "\n".join([f"// {s}\n{constants.START_SINGLE_PARSE_TOKEN}\n{p}\n{constants.END_SINGLE_PARSE_TOKEN}\n" for (s, p) in zip(query_sentences, query_parses)])

    p += condition_parse.strip() + "\n\n"
    p += query_parse.strip() + "\n\n" + constants.END_PARSE_TOKEN + "\n"
    return p


def parse(scenario, background_domains, experiment_dir, rng, args):
    # Construct a context for a given scenario.
    background_prompt = construct_background_domains_prompt(scenario, rng, background_domains, args)

    # Retrieve all of the sentences we plan to observe from the scenario.