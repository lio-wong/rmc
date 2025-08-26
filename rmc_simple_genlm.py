"""
Simple implementation of RMC that assumes that the 'translation transcript' is a sequentially growing set of alternating (τ, π) tuples.
Inputs: JSON stimulus file with a list of stimuli containing their background domains and observations.
Outputs: Translations of the stimuli into WebPPL code and inferences over the questions.
"""

import argparse
import json
import random
import numpy as np
import asyncio

import rmc.utils as utils

parser = argparse.ArgumentParser()

#### Experiment utilities
parser.add_argument('--base_dir', type=str, default="rmc-experiments",
    help='Base output directory for runs.')
parser.add_argument('--base_experiment_tag', type=str, default="demo",
    help='Base sub-dir for runs.')
parser.add_argument('--random_seed', type=int, default=0,
    help='Random seed')

### LLM parameters
parser.add_argument('--llm', type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    help='LLM to run')

### Experiment parameters.
parser.add_argument('--use_example_translations', type=bool, default=True, help='Use example translations.')
parser.add_argument('--base_background_prompt', type=str, help='Base background prompt.', default='prompts/rmc_simple_background_first_examples.txt')
parser.add_argument('--scenarios', type=str, help='Scenarios file.', default='scenarios/batches/demo_tow.json')
parser.add_argument('--num_scenarios', type=int, default=1, help='Number of scenarios to run.')


if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(args.random_seed)
    rng = np.random.default_rng(args.random_seed)

    experiment_tag, experiment_dir = utils.init_experiment_dir(
        base_dir=args.base_dir,
        base_experiment_tag=args.base_experiment_tag,
        llm_type=args.llm)
    # save the params to that folder
    with open(f"{experiment_dir}/params.json", "w") as f: json.dump(vars(args), f)

    # Simple RMC: base background system prompt.
    base_background_prompt = utils.read_file(args.base_background_prompt)
    print(f"Base background prompt: {base_background_prompt}")

    # Load the scenarios.
    scenarios = json.load(open(args.scenarios, "r"))

    from genlm.control import PromptedLLM, BoolCFG, AWRS
    
    for idx, scenario in enumerate(scenarios):
        if idx >= args.num_scenarios:
            break
        # For each scenario, get the domain prompt (we wont translate this yet.)
        scenario_background = utils.read_file(scenario["background"])
        print(f"#### PRINTING: Scenario {idx} background prompt\n\n####: {scenario_background}")

        llm = PromptedLLM.from_name(
            "meta-llama/Llama-3.2-1B-Instruct",
            eos_tokens=[b"<|eom_id|>", b"<|eot_id|>", b"\n\n"],
            temperature=0.5
        )
        conversation = [
            {"role": "system", "content": base_background_prompt + "\n" + scenario_background},
        ]
        if args.use_example_translations:
            for (sentence, translation) in scenario["example_question_translation"] + scenario["example_condition_translation"]:
                conversation.append({"role": "user", "content": "// " + sentence})
                conversation.append({"role": "assistant", "content": translation + "\n\n"})

        for item in conversation:
            print(f"{item['role']}: {item['content']}")

        # Extremely simple potential. Decoding from a prompt should start with condition, query, or var (for now).
        # TODO: for queries, if we start calling a variable, make sure it has the right named arguments.
        ppl_dsl = r"""
        start: expr
        expr: definition | condition | query

        definition: "var " MULTILINE_TEXT
        condition: "condition(" MULTILINE_TEXT
        query: "query: " MULTILINE_TEXT

        MULTILINE_TEXT: /.+/s

        %import common (NUMBER, ESCAPED_STRING, WS)
        %ignore WS      
        """
        ppl_cfg = BoolCFG.from_lark(ppl_dsl)

        from lark import Lark
        # Test the current CFG parser.
        test_parser = Lark(ppl_dsl)
        print(test_parser.parse("var x = 1"))
        print(test_parser.parse("query: intrinsic_strength_rank({athlete: 'Peyton', out_of_n_athletes: 100})"))
        print(test_parser.parse("condition(beat({team1: ['Taylor', 'Indiana'], team2: ['Quinn', 'Robin'], match: 5}))"))

        # Then for each scenario, translate the questions.
        for question in scenario["questions"]:
            conversation_item = {"role": "user", "content": "// " + question}
            print(f"{conversation_item['role']}: {conversation_item['content']}")
            conversation.append(conversation_item)
            llm.prompt_ids = llm.model.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=True,
                add_generation_prompt=True
            )
            # TODO: how does this work with the Chat interface?
            coerced_cfg = ppl_cfg.coerce(llm, f=b"".join)
            translation_token_sampler = AWRS(llm, coerced_cfg)
            sequences = asyncio.run(translation_token_sampler.smc(
                n_particles=2, # Number of candidate sequences to maintain
                ess_threshold=0.5, # Threshold for resampling
                max_tokens=500, # Maximum sequence length
                verbosity=1 # Print particles at each step
            ))
            sequences.decoded_posterior
            assert False
   

    # Then for each scenario, translate each observations and run inference after each translation (if we can) over the queries. -- TODO: figure out how to do this.
