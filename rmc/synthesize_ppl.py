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
    p += "\n" + constants.START_PARSE_TOKEN + "\n"
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
class WebPPLProgram():
    def __init__(self):
        self.definitions = []
        self.conditions = []
        self.queries = []

    def definitions_to_string(self):
        return "\n".join([f'\t// {fn_comment}\n{fn_def}' for (fn_comment, fn_def) in self.definitions]) + "\n"
    
    def conditions_to_string(self):
        return "\n".join([f'\t// {cond_comment}\n\t{cond}' for (cond_comment, cond) in self.conditions]) + "\n"
    
    def queries_to_string(self):
        query_comments = ""
        query_string = "return {\n"
        for q_comment, q in self.queries:
            query_comments += f"\t // {q_comment}\n"
            query_string += q + "\n"
        full_string = query_comments + "\n" + query_string + "\n" + "}"
        return full_string
        
    def to_string(self, posterior=True, 
                    posterior_samples=10000,
                    sampling_method='rejection'):
        model_str = "var model = function() {\n"

        model_str += "// BACKGROUND KNOWLEDGE\n"
        model_str += self.definitions_to_string()

        model_str += "// CONDITIONS\n"
        model_str += self.conditions_to_string()

        model_str += "//QUERIES"
        model_str += self.queries_to_string()
        model_str += "}\n"

        if posterior:
            sampling_string = f"var posterior = Infer({{ model: model, method: '{sampling_method}', samples: {posterior_samples} }});"
            if sampling_method == "MCMC":
                sampling_string = f"var posterior = Infer({{ model: model, method: '{sampling_method}', samples: {posterior_samples}, burn: 1000 }});"
            model_str += sampling_string
        return model_str

    def try_extend_potential_ppl_expression(self, nl_sentence, potential_ppl_expression, print_model=True, inference_timeout=30):
        sentence_expression = (nl_sentence, potential_ppl_expression)
        # Definition?
        if potential_ppl_expression.startswith(constants.WEBPPL_START_DEFINITION):
            self.definitions.append(sentence_expression)
        elif potential_ppl_expression.startswith(constants.WEBPPL_START_CONDITION):
            self.conditions.append(sentence_expression)
        elif potential_ppl_expression.startswith(constants.WEBPPL_START_QUERY):
            self.queries.append(sentence_expression)
        else:
            print("Sentence is not of any of the correct types!")
            return False
        
        model_str = self.to_string(posterior_samples=1)
        if print_model:
            print("====TRYING WEBPPL MODEL====")
            print(model_str)
            print("====TRYING WEBPPL MODEL====")
        key, sample_str = utils.run_webppl(code=model_str, timeout=inference_timeout)
        if key is None:
            print("ERROR, FAILED TO COMPILE")
            return False

        print("====SUCCESSFUL WEBPPL MODEL====")
        print(model_str)
        print("====SUCCESSFUL WEBPPL MODEL====")
        return True

class RMCModel(Model):
    def __init__(self, LLM, background_prompt, background_sentences, condition_sentences, query_sentences, max_tokens_per_step=500):
        super().__init__()
        self.LLM = LLM
        self.context = LMContext(LLM, background_prompt, show_prompt=False)
        self.background_sentences = background_sentences
        self.condition_sentences = condition_sentences
        self.query_sentences = query_sentences

        self.program = WebPPLProgram()
        self.remaining_sentences = background_sentences + condition_sentences + query_sentences
        self.max_tokens_per_step = max_tokens_per_step
        self.current_code_num_tokens = 0
        self.current_code_string = ""
        self.current_code_tokens = []

    def string_for_serialization(self):
        return f"{self.context}"

    async def step(self):
        print("================STARTING NEXT STEP============")
        if len(self.remaining_sentences) == 0:
            self.finish()
            return
        # Get the next sentence.
        next_sentence = self.remaining_sentences.pop(0)
        commented_next_sentence = f"// {next_sentence}\n{constants.START_SINGLE_PARSE_TOKEN}\n"

        # Intervene that the sentence is generated.
        commented_next_sentence_tokens = self.LLM.tokenizer.encode(commented_next_sentence)

        for comment_token in commented_next_sentence_tokens:
            await self.intervene(self.context.mask_dist(set([comment_token])), True)
            token = await self.sample(self.context.next_token())

        print("================CURRENT CONTEXT WITH SENTENCE============")
        print(self.string_for_serialization())
        print("================CURRENT CONTEXT WITH SENTENCE============")

        # Now generate until we have an expression block.
        # Reset code string.
        self.current_code_tokens = []
        self.current_code_string = ""
        self.current_code_num_tokens = 0
        while (not constants.END_SINGLE_PARSE_TOKEN in self.current_code_string) and self.current_code_num_tokens <= self.max_tokens_per_step:
            print(f"Sampling code token...; {self.current_code_tokens} tokens generated")
            token = await self.sample(self.context.next_token())
            self.current_code_tokens.append(token.token_id)
            self.current_code_string = f"{self.LLM.tokenizer.decode(self.current_code_tokens)}"
            self.current_code_num_tokens += 1
            print("================CURRENT CONTEXT WITH CODE============")
            print("CURRENT CODE_STRING:")
            print(self.current_code_string)
            print("================CURRENT CONTEXT WITH CODE============")

        # Extract the current WebPPL code.
        potential_code = self.current_code_string.split(constants.END_SINGLE_PARSE_TOKEN)[0]
        
        # Check if program is valid.
        self.condition(self.program.try_extend_potential_ppl_expression(nl_sentence=next_sentence, potential_ppl_expression=potential_code))
        

async def run_smc_async(LLM, background_prompt, background_sentences, condition_sentences, query_sentences, n_particles=1, ess_threshold=0.5):
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

    background_sentences = background_sentences[:2]
    condition_sentences = []
    query_sentences = []

    # Begin SMC-style parse. This could be generalized, for now we just assume a very stereotyped ordering of the vignettes.
    LLM = CachedCausalLM.from_pretrained(args.llm)
    particles = asyncio.run(run_smc_async(LLM, background_prompt, background_sentences, condition_sentences, query_sentences))

    # Currently its the same parse metdata for every particle
    parse_metadata = {
        "gen_metadata" : {
            "prompt": background_prompt,
            "background_domains" : tuple(background_domains)
        }
    }

    return particles, parse_metadata