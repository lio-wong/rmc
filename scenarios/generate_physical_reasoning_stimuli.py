### Generates physical reasoning stimuli from the physics language stimuli spreadsheet.
physical_reasoning_template_file = "scenarios/human-readable/physics_language_template.txt"
physical_reasoning_stimuli_file = "scenarios/human-readable/physics_language_stimuli.csv"

# Read in the template string.
with open(physical_reasoning_template_file, "r") as f:
    template = f.read()

# Read in the CSV.
import copy
import csv
import json

gold_parses = dict()
with open(physical_reasoning_stimuli_file, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        base_prompt = copy.deepcopy(template)
        conditions = [str(s).strip().replace(".", "") for s in [row['language_phrase_1'], row['language_phrase_2'], row['language_phrase_3'], row['language_phrase_4']] if len(str(s)) > 0]
        gold_parses_for_conditions = [str(s) for s in [row['code_phrase_1'], row['code_phrase_2'], row['code_phrase_3'], row['code_phrase_4']] if len(str(s)) > 0]

        conditions_string = "\n".join(conditions)
        base_prompt = base_prompt.replace("<CONDITIONS>", conditions_string)

        with open("scenarios/model-readable/physics/" + 'block-towers_' + row['task_id'] + ".txt", "w") as f:
            print('block-towers_' + row['task_id'] + ".txt")
            f.write(base_prompt)
    
        assert len(gold_parses_for_conditions) == len(conditions)
        for (c, p) in zip(conditions, gold_parses_for_conditions):
            gold_parses[c] = p

with open("prompts/condition-parses/block-towers.json", "w") as f:
    json.dump(gold_parses, f)