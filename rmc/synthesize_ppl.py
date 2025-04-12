import json
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

    scenario_text = utils.get_scenario_txt(scenario, args)
    condition_sentences = get_scenario_conditions(scenario_text)
    query_sentences = get_scenario_queries(scenario_text)

    # TODO: REPLACE WITH BACKGROUNDS FROM THE DOMAIN PROMPT.
    if args.replace_background_with_background_parses:
        background_parse_file = [b for b in args.background_parses if sports_domain in b][0]
        with open(os.path.join(args.background_domains_dir, background_parse_file + ".txt"), "r") as f:
            background_parse = f.read().strip()
        background_sentences = get_background_parse_sentences(background_parse)
    else:
        background_sentences = []
    return background_sentences, condition_sentences, query_sentences

def construct_background_domains_prompt(scenario, rng, background_domains, insert_into_raw_background_domain_str="", gold_parses=None, no_background_generation=False, args=None):
    if args.delimited_parse_generation:
       return construct_delimited_parse_generation_prompt(scenario, rng, background_domains, no_background_generation, args)
    else:
        # Code continuation format.
        return construct_code_continuation_prompt(scenario, rng, background_domains, no_background_generation, insert_into_raw_background_domain_str=insert_into_raw_background_domain_str, gold_parses=gold_parses, args=args)

def construct_code_continuation_prompt(scenario, rng, background_domains, no_background_generation, insert_into_raw_background_domain_str="", gold_parses=None, args=None):
    gold_program = None
    p = constants.CODE_CONTINUATION_HEADER
    p += "\n" + constants.LIBRARY_FUNCTIONS_HEADER + constants.LIBRARY_FUNCTIONS + "\n"

    shuffled_background_domains = rng.permutation(background_domains)
    if insert_into_raw_background_domain_str != "":
        with open(os.path.join(args.background_domains_dir, insert_into_raw_background_domain_str + ".txt"), "r") as f:
            background_parse = f.read().strip()
        assert no_background_generation # If we are using an existing background then there must be no background to generate.

        p += background_parse.split(constants.START_NEXT_CONDITIONS)[0]
        
        # Construct all of the conditions only for the continuation background domains.
       
        p += f"".join(construct_continuation_background_domain_prompt(background_domain, conditions_only=True, gold_parses=gold_parses, args=args)[0] for background_domain in shuffled_background_domains).strip()

        gold_program = construct_scenario_gold_program_from_raw_background_str(background_parse, scenario, gold_parses, args)
        return p, gold_program
    else:
        p += f"{constants.CODE_EXAMPLE_HEADER}\n\n".join(construct_continuation_background_domain_prompt(background_domain, args=args)[0] for background_domain in shuffled_background_domains)
        if no_background_generation:
            gold_string, gold_program = construct_continuation_background_domain_prompt(scenario, background_only=True, args=args)
            p += "\n\n" + constants.CODE_YOUR_EXAMPLE_HEADER + "\n" + gold_string
        else:
            p += "\n\n" + constants.CODE_YOUR_EXAMPLE_HEADER + "\n" + "var model = function() {\n"
    return p, gold_program

def construct_scenario_gold_program_from_raw_background_str(background_parse, scenario, gold_parses, args):
    example_program = WebPPLProgram(from_raw_background=background_parse)
    background_scenario_text = utils.get_scenario_txt(scenario, args)
    condition_sentences = get_scenario_conditions(background_scenario_text)
    condition_parses = [utils.gold_condition_parse(c, matches_to_teams={}, gold_parses=gold_parses) for c in condition_sentences]
    example_program.conditions =list(zip(condition_sentences, condition_parses))
    return example_program

def construct_delimited_parse_generation_prompt(scenario, rng, background_domains, no_background_generation, args):
    gold_program = None
    # Add in a header.
    if no_background_generation:
        print("Not yet implemented: no_background_generation")
        assert False
    p = constants.TRANSLATIONS_HEADER 

    # Add in library functions.
    p += "\n" + constants.LIBRARY_FUNCTIONS_HEADER + constants.LIBRARY_FUNCTIONS + "\n"

    shuffled_background_domains = rng.permutation(background_domains)
    p += "\n".join(construct_delimited_background_domain_prompt(background_domain, args) for background_domain in shuffled_background_domains)
    p += "\n" + constants.START_PARSE_TOKEN + "\n" 
    return p, gold_program

def construct_continuation_background_domain_prompt(background_domain, background_only=False, conditions_only=True, gold_parses=None, args=None):
    sub_domain = background_domain.split("_")[0]
    background_scenario_text = utils.get_scenario_txt(background_domain, args)
    # Construct a WebPPL model from the backgrounds and then write it out to a string.
    example_program = WebPPLProgram()

    if not conditions_only:
        background_parse_file = [b for b in args.background_parses if sub_domain in b][0]
        with open(os.path.join(args.background_domains_dir, background_parse_file + ".txt"), "r") as f:
            background_parse = f.read().strip()

        example_program.definitions = background_parse_file_to_definitions(background_parse)

    # Construct the conditions parse.
    condition_sentences = get_scenario_conditions(background_scenario_text)

    matches_to_teams = {}
    condition_parses = [utils.gold_condition_parse(c, matches_to_teams, gold_parses=gold_parses) for c in condition_sentences]
    # Construct the queries parse.
    if not conditions_only:
        query_sentences = get_scenario_queries(background_scenario_text)
        query_parses = [utils.gold_query_parse(query_sentence=q, sports_domain=sub_domain, query_idx=idx, matches_to_teams=matches_to_teams) 
                        for idx, q in enumerate(query_sentences)]
        example_program.queries = list(zip(query_sentences, query_parses))
    
    example_program.conditions =list(zip(condition_sentences, condition_parses))
    return example_program.to_string(include_expression_type_headers=False, background_only=background_only, conditions_only=conditions_only), example_program

def background_parse_file_to_definitions(background_parse):
    parses = background_parse.split("//")[1:]
    definition_tuples = []
    for p in parses:
        split_parse = p.split(constants.START_SINGLE_PARSE_TOKEN)
        sentence = split_parse[0].strip()
        code = split_parse[-1].split(constants.END_SINGLE_PARSE_TOKEN)[0].strip()
        definition_tuples.append((sentence, code))
    return definition_tuples

def construct_delimited_background_domain_prompt(background_domain, args):
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
    def __init__(self, from_raw_background=""):
        self.from_raw_background = ""
        self.raw_background_only = ""
        if len(from_raw_background) > 0:
            self.from_raw_background = from_raw_background
            self.raw_background_only = from_raw_background.split(constants.START_NEXT_CONDITIONS)[0]
            self.raw_queries_only = from_raw_background.split(constants.START_NEXT_CONDITIONS)[-1]
        self.definitions = []
        self.conditions = []
        self.queries = []

    def definitions_to_string(self):
        if len(self.from_raw_background) > 0:
            return self.raw_background_only
        return "\n\n".join([f'\t// {fn_comment}\n\t{fn_def}' for (fn_comment, fn_def) in self.definitions]) + "\n\n"
    
    def conditions_to_string(self):
        return "\n\n".join([f'\t// {cond_comment}\n\t{cond}' for (cond_comment, cond) in self.conditions]) + "\n\n"
    
    def queries_to_string(self):
        if len(self.from_raw_background) > 0:
            return self.raw_queries_only
        query_string = "return {\n"
        for q_comment, q in self.queries:
            query_string += f"\t // {q_comment}\n\t{q},\n"
        query_string += "// END OF QUERIES\n" # Necessary to terminate the queries for extraction so we can re-add in the return.
        query_string += "}\n"
        return query_string + "\n\n"
        
    def to_string(self, posterior=True, 
                    posterior_samples=10000,
                    sampling_method='rejection', include_expression_type_headers=True,
                    background_only=False, conditions_only=False):
        
        model_str = "var model = function() {\n"

        if include_expression_type_headers:
            model_str += "// BACKGROUND KNOWLEDGE\n"
        model_str += self.definitions_to_string()

        # Helper method to generate up to this point for synthesis prompting.
        if background_only:
            return model_str

        if include_expression_type_headers:
            model_str += "// CONDITIONS\n"
        model_str += self.conditions_to_string()

        if conditions_only:
            return self.conditions_to_string()
        
        if include_expression_type_headers:
            model_str += "//QUERIES\n"

        model_str += "// RETURN INFERENCE RESULTS\n" # Keep this to have a comment.
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
            # Remove any trailing comma.
            if sentence_expression[1].endswith(","):
                sentence_expression = (sentence_expression[0], sentence_expression[1][:-1])
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
        return True

class RMCModel(Model):
    def __init__(self, LLM, background_prompt, background_sentences, condition_sentences, query_sentences, max_tokens_per_step=500, delimited_parse_generation=True, no_background_generation=False, no_condition_generation=False, temperature=0.2, args=None, gold_program=None):
        super().__init__()
        self.LLM = LLM
        self.context = LMContext(LLM, background_prompt, show_prompt=False, temp=temperature)
        print(f"==Initialized context with temperature: {self.context.temp}\n")
        self.background_sentences = background_sentences
        self.condition_sentences = condition_sentences
        self.query_sentences = query_sentences
        self.delimited_parse_generation = delimited_parse_generation

        self.program = WebPPLProgram()
        if no_background_generation:
            self.program.definitions = gold_program.definitions

        if no_background_generation:
            self.remaining_sentences = condition_sentences + query_sentences
        else:
            self.remaining_sentences = background_sentences + condition_sentences + query_sentences
        self.max_tokens_per_step = max_tokens_per_step
        self.current_code_num_tokens = 0
        self.current_code_string = ""
        self.current_code_tokens = []

    def string_for_serialization(self):
        return f"{self.context}"

    async def step(self):
        # TODO: HANDLE THE QUERY GENERATION.
        # TODO: there are newline characters in the code lol.

        print("================STARTING NEXT STEP============")
        print(f"Now remaining: {len(self.remaining_sentences)} sentences.")
        if len(self.remaining_sentences) == 0:
            self.finish()
            return
        # Get the next sentence.
        next_sentence = self.remaining_sentences.pop(0)

        # If we're not in delimited parse generatoin mode, we need to generate the return query before queries.
        if len(self.remaining_sentences) == (len(self.query_sentences) - 1):
            if not self.delimited_parse_generation:
                return_query = "\n// RETURN INFERENCE RESULTS\nreturn {\n"
                return_query_tokens = self.LLM.tokenizer.encode(return_query)
                for return_query_token in return_query_tokens:
                    await self.intervene(self.context.mask_dist(set([return_query_token])), True)
                    token = await self.sample(self.context.next_token())
            print("================CURRENT CONTEXT WITH SENTENCE============")
            print(self.string_for_serialization())
            print("================CURRENT CONTEXT WITH SENTENCE============")

        if self.delimited_parse_generation:
            commented_next_sentence = f"// {next_sentence}\n{constants.START_SINGLE_PARSE_TOKEN}\n"
        else:
            commented_next_sentence = f"// {next_sentence}\n"

        # Intervene that the sentence is generated.
        commented_next_sentence_tokens = self.LLM.tokenizer.encode(commented_next_sentence)

        for comment_token in commented_next_sentence_tokens:
            await self.intervene(self.context.mask_dist(set([comment_token])), True)
            token = await self.sample(self.context.next_token())


        # Now generate until we have an expression block.
        # Reset code string.
        self.current_code_tokens = []
        self.current_code_string = ""
        self.current_code_num_tokens = 0

        # Note that this currently assumes no commenting in the code.
        stop_token = constants.END_SINGLE_PARSE_TOKEN if self.delimited_parse_generation else "//"
        while (not stop_token in self.current_code_string) and self.current_code_num_tokens <= self.max_tokens_per_step:
            if (len(self.remaining_sentences) <= len(self.query_sentences)):
                print("================CURRENT CONTEXT WITH CODE============")
                print(self.string_for_serialization())
                print("================CURRENT CONTEXT WITH CODE============")

            token = await self.sample(self.context.next_token())
            self.current_code_tokens.append(token.token_id)
            self.current_code_string = f"{self.LLM.tokenizer.decode(self.current_code_tokens)}"
            self.current_code_num_tokens += 1
            if self.current_code_num_tokens % 100 == 0:
                print(f"Sampling code token: {self.current_code_num_tokens} / {self.max_tokens_per_step} max tokens generated")
                print("================CURRENT CONTEXT WITH CODE============")
                print("SENTENCE TO PARSE:")
                print(str(commented_next_sentence))
                print("CURRENT CODE_STRING:")
                print(self.current_code_string)
                print("================CURRENT CONTEXT WITH CODE============")
            
        # Extract the current WebPPL code.
        potential_code = self.current_code_string.split(stop_token)[0].strip()
        
        # Check if program is valid.
        self.condition(self.program.try_extend_potential_ppl_expression(nl_sentence=next_sentence, potential_ppl_expression=potential_code))
        
async def run_smc_async(LLM, background_prompt, background_sentences, condition_sentences, query_sentences, delimited_parse_generation=False, n_particles=1, no_background_generation=False, no_condition_generation=False, ess_threshold=0.5, gold_program=None, args=None):
    # Cache the key value vectors for the prompt.
    LLM.cache_kv(LLM.tokenizer.encode(background_prompt))

    # Initialize the model.
    rmc_model = RMCModel(LLM, background_prompt, background_sentences, condition_sentences, query_sentences, delimited_parse_generation=delimited_parse_generation, no_background_generation=no_background_generation, no_condition_generation=no_condition_generation, gold_program=gold_program, temperature=args.parsing_temperature)
    
    # Run inference
    print(f"Initializing SMC with {n_particles} particles.")
    # TODO: replace NONE with output reading.
    particles = await smc_standard(
        rmc_model, n_particles, ess_threshold, None, None
    )
    return particles


def parse(scenario, background_domains, insert_into_raw_background_domain_str, experiment_dir, rng, args):
    # Construct a context for a given scenario.
    if len(str(args.gold_parses)) > 0:
        gold_parses = json.load(open(args.gold_parses))
    else:
        gold_parses = None
    background_prompt, gold_program = construct_background_domains_prompt(scenario, rng, background_domains, no_background_generation=args.no_background_generation, insert_into_raw_background_domain_str=insert_into_raw_background_domain_str, gold_parses=gold_parses,args=args)

    

    print("========BACKGROUND DOMAINS:==========")
    print(background_domains)
    print("========BACKGROUND PROMPT:==========")
    print(background_prompt)

    # Retrieve all of the sentences we plan to observe from the scenario.
    background_sentences, condition_sentences, query_sentences = get_all_scenario_sentences(scenario, args)
    
    # Begin SMC-style parse. This could be generalized, for now we just assume a very stereotyped ordering of the vignettes.
    LLM = CachedCausalLM.from_pretrained(args.llm, engine_opts={
        "max_model_len" : 10000 # Context window length. Set to reduce memory.
    })
    particles = asyncio.run(run_smc_async(LLM, background_prompt, background_sentences, condition_sentences, query_sentences, n_particles=args.number_of_particles, delimited_parse_generation=args.delimited_parse_generation, no_background_generation=args.no_background_generation, no_query_generation=args.no_query_generation, gold_program=gold_program, args=args))

    # Currently its the same parse metdata for every particle
    parse_metadata = {
        "gen_metadata" : {
            "prompt": background_prompt,
            "background_domains" : tuple(background_domains)
        }
    }

    return particles, parse_metadata