import os
import sys
import asyncio
sys.path.append("..")
sys.path.append("../hfppl")

from rmc import constants, utils
from hfppl import CachedCausalLM
from hfppl import LMContext
from hfppl import Model
from hfppl import smc_standard


def get_scenario_conditions(scenario):
    condition_sentences = scenario.split('CONDITIONS')[1].split('QUERIES')[0].split("\n")
    condition_sentences = [s.strip() for s in condition_sentences if len(s.strip()) > 0]
    # remove ending periods if there (b/c not always generated + leading to parse issues)
    condition_sentences = [s[:-1] if s[-1] == "." else s for s in condition_sentences]
    return condition_sentences

def get_background_parse_sentences(background_parse):
    return [a.split("// ")[-1].strip() for a in background_parse.split(constants.START_SINGLE_PARSE_TOKEN)][:-1]

def get_scenario_queries(scenario):
    query_sentences = scenario.split('QUERIES')[1].split("\n")
    query_sentences = [s.strip() for s in query_sentences if len(s.strip()) > 0 and not s.startswith("//")]
    return query_sentences


def get_all_scenario_sentences(scenario, args):
    sports_domain = scenario.split("_")[0]
    background_parse_file = [b for b in args.background_parses if sports_domain in b][0]
    with open(os.path.join(args.background_domains_dir, background_parse_file + ".txt"), "r") as f:
        background_parse = f.read().strip()

    scenario_text = utils.get_scenario_txt(scenario, args)
    condition_sentences = get_scenario_conditions(scenario_text)
    query_sentences = get_scenario_queries(scenario_text)

    # TODO: REPLACE WITH BACKGROUNDS FROM THE DOMAIN PROMPT.
    if args.replace_background_with_background_parses:
        background_sentences = get_background_parse_sentences(background_parse)
    else:
        assert False
    return background_sentences, condition_sentences, query_sentences

def construct_background_domains_prompt(scenario, rng, background_domains, args):
    # Add in a header.
    p = constants.TRANSLATIONS_HEADER 

    # Add in library functions.
    p += "\n" + constants.LIBRARY_FUNCTIONS_HEADER + constants.LIBRARY_FUNCTIONS + "\n"

    shuffled_background_domains = rng.permutation(background_domains)
    p += "\n".join(construct_background_domain_prompt(background_domain, args) for background_domain in shuffled_background_domains)
    p += "\n" + constants.END_PARSE_TOKEN + "\n"
    return p

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

#### SMC-RMC 
class RMCModel(Model):
    def __init__(self, LLM, background_prompt, background_sentences, condition_sentences, query_sentences):
        super().__init__()
        self.LLM = LLM
        self.context = LMContext(LLM, background_prompt)
        self.background_sentences = background_sentences
        self.condition_sentences = condition_sentences
        self.query_sentences = query_sentences

        self.program = ""
        self.remaining_sentences = background_sentences + condition_sentences + query_sentences

    async next_tokens(sentence):
        tokens = self.LLM.tokenizer()

    async def step(self):
        if len(self.remaining_sentences) == 0:
            self.finish()
            return
        next_sentence = self.remaining_sentences.pop()
        continuation = f"// {next_sentence}\n{constants.START_SINGLE_PARSE_TOKEN}\n"
        print(next_sentence)
        # Intervene the sentences
        await self.intervene()
   


async def run_smc_async(LLM, background_prompt, background_sentences, condition_sentences, query_sentences, n_particles=2, ess_threshold=0.5):
    # Cache the key value vectors for the prompt.
    LLM.cache_kv(LLM.tokenizer.encode(background_prompt))

    # Initialize the Model
    rmc_model = RMCModel(LLM, background_prompt, background_sentences, condition_sentences, query_sentences)
    
    # Run inference
    # TODO: replace NONE with output reading.
    particles = await smc_standard(
        rmc_model, n_particles, ess_threshold, None, None
    )

    return particles


def parse(scenario, background_domains, experiment_dir, rng, args):
    # Construct a context for a given scenario.
    background_prompt = construct_background_domains_prompt(scenario, rng, background_domains, args)

    # Retrieve all of the sentences we plan to observe from the scenario.
    background_sentences, condition_sentences, query_sentences = get_all_scenario_sentences(scenario, args)

    # Begin SMC-style parse. This could be generalized, for now we just assume a very stereotyped ordering of the vignettes.
    LLM = CachedCausalLM.from_pretrained(args.llm)
    particles = asyncio.run(run_smc_async(LLM, background_prompt, background_sentences, condition_sentences, query_sentences))