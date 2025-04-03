from hmac import new
import importlib
import c
from matplotlib.pylab import plt 
import seaborn as sns 
import numpy as np
import os
import json
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from openai import models
from matplotlib.gridspec import GridSpec
from PIL import Image
import os
import re

def process_query_values(queries_list, n_queries, full_key=None, method='flat'):
    """
    Helper function to process a list of queries into a flat dictionary of values.
    """
    flat_vals = {query_idx: [] for query_idx in range(n_queries)}
    
    for queries in queries_list:
        if queries is None:
            continue
            
        if full_key is not None:
            queries = queries[full_key]
            
        for query, values_list in queries.items():
            query_idx = int(query.split("query")[-1])
            if method == 'mean': 
                flat_vals[query_idx-1].append(np.mean(values_list))
            else: flat_vals[query_idx-1].extend(values_list)
            
    return flat_vals

def process_values_from_all_sources(all_scenarios, llm_res, baseline_human_data, 
                                  all_inference_data, gold_inference_data, n_queries, 
                                  method='flat', skip_dive=True):
    """
    Process and aggregate values from all data sources into a structured dictionary.
    """
    processed_vals_all_srcs = {scenario: {} for scenario in all_scenarios}

    for scenario in all_scenarios:
        if skip_dive and 'diving' in scenario:
            continue
            
        # Process LLM results (both vanilla and CoT)
        for llm_type in ['vanilla', 'cot']:
            processed_vals_all_srcs[scenario][llm_type] = process_query_values(
                llm_res[llm_type][scenario], n_queries,method=method
            )
        
        # Process baseline human data
        processed_vals_all_srcs[scenario]['human_no_commentary'] = process_query_values(
            baseline_human_data[scenario], n_queries,method=method
        )
        
        # Process gold inference data if available
        if gold_inference_data is not None and scenario in gold_inference_data:
            processed_vals_all_srcs[scenario]['gold'] = process_query_values(
                gold_inference_data[scenario], n_queries, full_key='full',method=method
            )
            
        # Process synthesis pipeline data if available
        if scenario in all_inference_data:
            processed_vals_all_srcs[scenario]['model_full'] = process_query_values(
                all_inference_data[scenario], n_queries, full_key='full',method=method
            )

    return processed_vals_all_srcs


def html_summary(experiment_dir):
    analysis_dir = os.path.join(experiment_dir, "analyses")
    synthesis_metadata_path = os.path.join(analysis_dir, "synthesis_metadata.json")
    if not os.path.exists(analysis_dir): 
        os.makedirs(analysis_dir)
    if not os.path.exists(synthesis_metadata_path):
        # Construct the JSONs.
        inference_results = [os.path.join(experiment_dir, f) for f in os.listdir(experiment_dir) if f.startswith("inference_results_")]
        json_results = []
        for f in inference_results:
            j = json.load(open(f))
            json_results.append(j)  
        with open(synthesis_metadata_path, "w") as f:
            json.dump(json_results, f)

    print("=== HTML summary ==== \n")
    print("If you haven't already: launch server with python -m http.server \n")
    print(f"Open in browser: http://localhost:8080/analysis/html/scenario_summary.html?path=../../{synthesis_metadata_path}")

def get_stimuli_ids(scenario_dir="data/scenarios/june-pilot-races/", use_pilot_filtering=True): 
    all_scenarios = [f"{scenario_dir}{stim.split('.txt')[0]}" for stim in os.listdir(f"{scenario_dir}")]
    print("SCENARIOS: ",len(all_scenarios))
    if use_pilot_filtering: 
        # stimuli_ids = [s.split("/")[-1] for s in all_scenarios if "confounded-with-partner" in s or "team-consistent-long" in s or "team-consistent-short" in s or "individ-consistent" in s]
        stimuli_ids = [s.split("/")[-1] for s in all_scenarios if "effort" in s] #or "tired" in s]
        
    else: 
        stimuli_ids = [s.split("/")[-1] for s in all_scenarios]
    print("\n\nFILTERED SCENARIOS: ", len(stimuli_ids))
    return stimuli_ids, all_scenarios

def load_ensemble_res(dir): 
    try: 
        with open(f"{dir}/ensembled_res.json", "r") as f: 
            ensemble_res = json.load(f)
    except:
        ensemble_res = None
    return ensemble_res

def compute_se(vals): 
    return 1.96 * np.std(vals)/(len(vals) ** 0.5)

def compute_wd(vals1, vals2): 
    return wasserstein_distance(vals1, vals2)

def z_score_transform(values):
    # Convert the list to a numpy array
    values_array = np.array(values)
    
    # Calculate the mean and standard deviation of the values
    mean = np.mean(values_array)
    std_dev = np.std(values_array)
    
    # Perform the z-score transformation
    z_scores = (values_array - mean) / std_dev
    
    return z_scores

def process_gold_resps(gt_human_data, sorted_stimuli_ids): 
    humans = ["gold_response_human_katie", "gold_response_human_tyler", "gold_response_human_lance", "gold_response_human_ced"]
    gt_human_agg_res = {f"{stimuli_id}.txt": {} for stimuli_id in sorted_stimuli_ids}
    for stimuli_id in sorted_stimuli_ids: 
        stimuli_id += ".txt"
        stim_df = gt_human_data.loc[gt_human_data["stimuli_id"] == stimuli_id]
        h_resps = []
        r_katie = stim_df["gold_response_human_katie"].iloc[0]
        r_tyler = stim_df["gold_response_human_tyler"].iloc[0]
        r_lance = stim_df["gold_response_human_lance"].iloc[0]
        r_ced = stim_df["gold_response_human_ced"].iloc[0]
        if str(r_katie) != "nan": 
            h_resps.append(r_katie)
        # else: continue 
        if str(r_tyler) != "nan":
            h_resps.append(r_tyler)
        # else: continue 
        if str(r_lance) != "nan":
            h_resps.append(r_lance)
        # else: continue 
        if str(r_ced) != "nan":
            h_resps.append(r_ced)
        # else: continue 
        if len(h_resps) == 0: 
            # print(stimuli_id)
            continue

        if len(h_resps) > 1: 
            diff = np.abs(h_resps[0] - h_resps[1])
            #diff2 = np.abs(h_resps[0] - h_resps[2])
            if diff > 20: print(stimuli_id, h_resps)


        gt_human_agg_res[stimuli_id] = {"agg": np.mean(h_resps), "se": compute_se(h_resps), "all": h_resps}
    return gt_human_agg_res

def compare_raw_gold(human_agg_res, gt_human_agg_res, verbose=False):
    # compare prolific resps [human_agg_res] with gold resps [gt_human_agg_res]
    prolific_resps = []
    gold_resps = []
    for stim_id in gt_human_agg_res.keys(): 
        if "agg" in gt_human_agg_res[stim_id]:
            gt_resps = gt_human_agg_res[stim_id]["agg"]
            new_resps = human_agg_res[stim_id]["agg"]
            if np.abs(gt_resps - new_resps) > 20: 
                if verbose: print(stim_id, gt_resps, new_resps)
            prolific_resps.append(new_resps)
            gold_resps.append(gt_resps)
        else: print("missing: ", stim_id)
    return pearsonr(prolific_resps, gold_resps)



def process_llm_baselines(sorted_stimuli_ids): 
    # baseline_data = pd.read_csv("marshalling_baselines_results_new.csv")
    # baseline_data_llama = pd.read_csv("marshalling_baselines_results_new_llama3.csv")
    baseline_data = pd.read_csv("marshalling_baselines_results_gpt.csv")
    baseline_data_llama = pd.read_csv("marshalling_baselines_results_llama.csv")

    baseline_types = ["vanilla", "cot"]
    baseline_models = ['gpt-4o-2024-05-13', 'meta-llama/Llama-3-70b-chat-hf']
                    #,'gpt-3.5-turbo-0125', 'meta-llama/Llama-3-70b-chat-hf',
        #    'meta-llama/Llama-3-8b-chat-hf','mistralai/Mixtral-8x22B-Instruct-v0.1',]

    stim2baselines = {f"{stim}.txt": {} for stim in sorted_stimuli_ids}
    stim2baselines_rationale = {f"{stim}.txt": {} for stim in sorted_stimuli_ids}

    for stim in sorted_stimuli_ids: 
        stim += ".txt"
        stim_entry = baseline_data.loc[baseline_data["stimuli_id"] == stim] #== stim.split(".txt")[0]]
        for baseline_type in baseline_types: 
            for baseline_model in baseline_models: 
                    model_tag = f"{baseline_model}_{baseline_type}"
                    if "meta" in baseline_model: 
                        stim_entry = baseline_data_llama.loc[baseline_data_llama["stimuli_id"] == stim]
                        stim2baselines[stim][model_tag] = stim_entry[model_tag].values[0]
                        if "cot" in model_tag: stim2baselines_rationale[stim][model_tag] = stim_entry[model_tag + "_rationale"].values[0]
                    else: # hack for now -- revert to just this
                        stim_entry = baseline_data.loc[baseline_data["stimuli_id"] == stim] #== stim.split(".txt")[0
                        stim2baselines[stim][model_tag] = stim_entry[model_tag].values[0]
                        if "cot" in model_tag: stim2baselines_rationale[stim][model_tag] = stim_entry[model_tag + "_rationale"].values[0]
                    
    return stim2baselines, stim2baselines_rationale 



def process_human_data(human_data, sorted_stimuli_ids): 
    human_z_scores = []
    participant_ids = human_data["prolific_id"].unique()
    human_data["norm_slider_values"] = [0 for _ in range(len(human_data))]


    for pid in participant_ids: 
        participant_df = human_data.loc[human_data["prolific_id"] == pid]
        
        # todo: fix z-score
        # filter to rows where "How fast" is in the "question" column 
        how_fast_df = participant_df.loc[participant_df["question"].str.contains("faster than")]

        score_idxs = list(participant_df.index)
        raw_scores = list(participant_df["slider_val"].values)
        norm_scores = z_score_transform(raw_scores)
        # print("Orig scores: ", raw_scores, "\n\tNormalized scores:", norm_scores)
        human_z_scores.append(norm_scores)
        for idx, score in zip(score_idxs, norm_scores): 
            human_data.at[idx, "norm_slider_values"] = score


    question_df = human_data.loc[human_data["question"].str.contains("How fast")]
    score_idxs = list(question_df.index)
    raw_scores = list(question_df["slider_val"].values)
    for idx, score in zip(score_idxs, norm_scores): 
        human_data.at[idx, "norm_slider_values"] = score

    question_df = human_data.loc[human_data["question"].str.contains("How likely")]
    score_idxs = list(question_df.index)
    raw_scores = list(question_df["slider_val"].values)
    for idx, score in zip(score_idxs, norm_scores): 
        human_data.at[idx, "norm_slider_values"] = score
        
    stimui_ids_human = []
    tag2questions = {}
    # team-consistent-short-variant-5*Is_Robin_faster_than_a_typical_runner?.txt
    for _, entry in human_data.iterrows(): 
        tag = entry.tag 
        question = entry.question.replace(" ", "_")
        
        question = question.replace("random_", "")
        if tag in tag2questions: 
            tag2questions[tag].add(question)
        else:
            tag2questions[tag] = {question} 
        stim_id=f"{tag}*{question}.txt"
        stimui_ids_human.append(stim_id)
    human_data["stimuli_id"] = stimui_ids_human

    human_agg_res = {}
    for stim_id in sorted_stimuli_ids:
        stim_id += ".txt"
        stim_df = human_data.loc[human_data["stimuli_id"] == stim_id]
        resps = list(stim_df["slider_val"])
        #resps = list(stim_df["norm_slider_values"])
        agg = np.mean(resps)

        human_agg_res[stim_id] = {"agg": np.mean(resps), "median": np.median(resps), "all": resps, "se": compute_se(resps)}
        
    return human_data, human_agg_res, tag2questions


def process_model_results(main_res_dir, sorted_stimuli_ids, filetag= "_cond_tug_of_warMatchesCompare_jump_2_"):
        
    scenario2full_res = {}
    all_res_data = {"stimuli_id": [], "tag": [], "scenario_type": [], "question": [], "mean_ensemble_model": [], "std_ensemble_model": [], 
    # "mean_model": [], "std_model": [], 
    "compile_rate": [],
    "model_scores": [], 
    "model_means": [],
    # "best_model": [], "worst_model": []
    "all_samples": []
    }

    valid_semantic_scores = []
    valid_per_defn_semantic_scores = []

    #for res_dir in dev_res: 
        #scenario, ablation, question = res_dir.split("*")
    for stimuli_id in sorted_stimuli_ids: 
        # stimuli_id = stimuli_id.split("/")[-1]
        scenario, question = stimuli_id.split("*")
        scenario_variant = scenario.split("-")[-1]
        scenario = "-".join(scenario.split("-")[:-1])
        question = question.split(".txt")[0]
        res_dir = f'{scenario}-{scenario_variant}*{question}{filetag}'
        res_dir = res_dir.replace(" ", "_")
        # question = question.split("_cond")[0]
        stimuli_res_dir = f"{main_res_dir}{res_dir}"
        res_data = load_ensemble_res(stimuli_res_dir)
        
        if res_data is None: 
            print("MISSING: ", res_dir)
            continue 
            # missing_stims.add(stimuli_id)
            # all_res_data["mean_model"].append('null')
            # all_res_data["std_model"].append('null')
            # all_res_data["compile_rate"].append(-1)
        else: 
            all_res_data["tag"].append(scenario) 
            all_res_data["scenario_type"].append(scenario_variant.upper())
            #question = question.replace('_', ' ')
            all_res_data["question"].append(question)
            # print("STIM: ", f"{scenario}-{scenario_variant}*{question}.txt")
            all_res_data["stimuli_id"].append(f"{scenario}-{scenario_variant}*{question}.txt")
            scenario2full_res[f"{scenario}-{scenario_variant}*{question}"] = res_data
            post_summary = res_data["post_summary"]
            executability = [x if x == 1 else 0 for x in res_data["executability_per_model"]] # convert so mean isn't impacted by -1
            semantic_scores = [x for x in res_data["semantic_scores_per_model"] if x != -1]
            valid_per_defn_semantic_scores.extend([list(x.values()) for x in res_data["semantic_per_defn_per_model"] if len(x) != 0])
            valid_semantic_scores.append(semantic_scores)
            if post_summary != {}:

                all_res_data["mean_ensemble_model"].append(res_data["post_summary"]['mean'])
                all_res_data["std_ensemble_model"].append(res_data["post_summary"]['standard_deviation'])
                all_res_data["model_scores"].append(res_data["scores_per_model"])
                
                # NOTE -- we saved out individ summaries  for the "how fast" question using the unnormalized
                # so need to normalize.... 
                
                model_means = [] 
                all_post_samples = []
                valid_res = res_data["individ_summaries"].keys()
                for idx in range(10): # todo -- change when we have a different number of rollouts... 
                    if str(idx) not in valid_res: continue
                    with open(f"{stimuli_res_dir}/llm-ppl_rollout{idx}.json") as f: 
                        raw_res = json.load(f)
                    
                    try: 
                        
                        prior_samples = eval(raw_res["official_prior_samples"])
                        post_samples = eval(raw_res["official_posterior_samples"])
                        
                        post_mean = np.mean([x["value"] for x in post_samples]) * 100
                        model_means.append(post_mean)#updated_post_summary["mean"])
                        all_post_samples.append(post_samples)
                    except Exception as e:
                        print("error: ", e) 
                        continue 
                
                # else:                 
                    
                #     for _, entry in res_data["individ_summaries"].items(): 
                #         model_means.append(entry["post"]["mean"])
                all_res_data["model_means"].append(model_means)
                all_res_data["all_samples"].append(all_post_samples)
            else: 
                all_res_data["mean_ensemble_model"].append('null')
                all_res_data["std_ensemble_model"].append('null')
                all_res_data["model_scores"].append('null')
                all_res_data["model_means"].append('null')
                all_res_data["all_samples"].append('null')
            all_res_data["compile_rate"].append(np.mean(executability))


    res_df = pd.DataFrame.from_dict(all_res_data)
        

    return res_df



def compute_pval_viz(pval): 
    if pval < 0.01:
        stat_sig = "(***)"
    elif pval < 0.05: 
        stat_sig = "(**)"
    elif pval < 0.1: 
        stat_sig = "(*)"
    else: stat_sig = ""
    return stat_sig

def process_corr(corr): 
    corr_val, p_val = corr
    corr_val = round(corr_val, 2)
    stat_sig_viz = compute_pval_viz(p_val)
    return f"{corr_val} {stat_sig_viz}"


def visualize_res(comp_model, comp_model_worst, comp_baselines, comp_human, comp_human_se, comp_human_all, question_types, models_viz, question_type2viz, ):

    fig, axes = plt.subplots(1,2,figsize=(14,6))

    question_classes = question_type2viz.keys()
    
    # colors = {"best_score": "blue", "worst_score": "red", "opt_score": "purple"}
    all_corrs = {q: {m: None for m in models_viz} for q in question_classes}
    all_rank_corrs = {q: {m: None for m in models_viz} for q in question_classes}
    # all_wd = {q: {m: None for m in models_viz} for q in question_classes}


    show_err = True 
    for i, question_type in enumerate(question_classes):#["how_likely","how_fast",]): 
        ax = axes[i]
        for j, score_type in enumerate(models_viz): 
            color = f"C{j}"
            if score_type == "best_score": 
                model_preds = [x[0] for x in comp_model] 
                comp_model_se = [x[1] for x in comp_model]
            elif score_type == "worst_score": 
                model_preds = [x[0] for x in comp_model_worst]
                comp_model_se = [x[1] for x in comp_model_worst]
            elif score_type in comp_baselines:
                full_model_preds = comp_baselines[score_type]
                full_model_preds = [eval(m) for m in full_model_preds]
                model_preds = [np.mean(m) for m in full_model_preds]
                comp_model_se = np.array([compute_se(m) for m in full_model_preds])
            
            comp_model_sub = np.array(model_preds)[question_types == question_type]
            comp_human_sub = np.array(comp_human)[question_types == question_type]
            
            pcorr = pearsonr(comp_human_sub, comp_model_sub)
            scorr = spearmanr(comp_human_sub, comp_model_sub)
            all_corrs[question_type][score_type] = pcorr
            all_rank_corrs[question_type][score_type] = scorr
            
            # print(question_type, score_type, "\n\t", pearsonr(comp_human_sub, comp_model_sub), spearmanr(comp_human_sub, comp_model_sub), len(comp_model_sub))
            if j == 0: print(question_type, len(comp_model_sub))

            comp_human_se_sub = np.array(comp_human_se)[question_types == question_type]
            
            
            # ax.scatter(comp_model_sub, comp_human_sub, alpha=0.5, 
            #            label=question_type2viz[question_type], 
            #            s=80,color=colors[i])# yerr=comp_human_se_sub)
            
            #if score_type != "none": 
            if (score_type == "best_score") or (("gpt-4" in score_type or "llama" in score_type) and "cot" in score_type): 
                sns.regplot(x=comp_model_sub, y=comp_human_sub, ax=ax, color=color, scatter=True,
                            label=score_type, ci=95)
                if show_err: 
                    ax.errorbar(comp_model_sub, comp_human_sub,
                            yerr = comp_human_se_sub,
                            color=color,
                            alpha=0.3,
                            #fmt ='o',
                        fmt='none', capsize=5, zorder=1) #color='C0')
                # if score_type in comp_baselines: 
                # print("err: ",np.array(comp_model_se)[question_types == question_type] )
                ax.errorbar(comp_model_sub, comp_human_sub,
                        xerr = np.array(comp_model_se)[question_types == question_type],
                        color=color,
                        alpha=0.3,
                        #fmt ='o',
                    fmt='none', capsize=5, zorder=1)
            
        lims = [
            -5, 105
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)


        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel("Model Prediction", fontsize=18)
        ax.set_ylabel("Human Prediction", fontsize=18)
        ax.set_title(question_type2viz[question_type], fontsize=20)
        if i == 1: ax.legend(loc='upper center', bbox_to_anchor=(-0.2, -0.15),
                ncol=3, fancybox=True, shadow=True, fontsize=14)
            
    return all_corrs
            

def generate_res_table(corr_data, question_type2viz, models_viz): 
    df_data = {'Model Class': [], 'Model': [], 'Event Likelihood': [], 'Speed Inferences': [],}

    # Populate the DataFrame dictionary
    for model in models_viz: 
        if model in ['best_score', 'worst_score']:#, 'opt_score']:
            model_class = 'Ours'
            model_variant = model.split("_")[0] + " score" if model != 'opt_score' else 'Opt Selection'
        else:
            model_class = model.split('_')[0]
            model_variant = model.split("_")[-1]
        
        df_data['Model Class'].append(model_class)
        df_data['Model'].append(model_variant)
        corr_val, p_val = corr_data['how_fast'][model]
        df_data[question_type2viz["how_fast"]].append(process_corr(corr_data['how_fast'][model]))
        df_data[question_type2viz["how_likely"]].append(process_corr(corr_data['how_likely'][model]))

    # Create the DataFrame
    df = pd.DataFrame(df_data)

    # Sort the DataFrame by 'Model Class' and then by 'Model'
    # df = df.sort_values(by=['Model Class', 'Model'])
    # Define a custom sort order
    model_class_order = ['gpt-4o-2024-05-13', 'meta-llama/Llama-3-70b-chat-hf', 'Ours']#, 'gpt-3.5-turbo-0125',  'meta-llama/Llama-3-8b-chat-hf', 'mistralai/Mixtral-8x22B-Instruct-v0.1', 'Ours']
    df['Model Class'] = pd.Categorical(df['Model Class'], categories=model_class_order, ordered=True)
    #df['Model'] = pd.Categorical(df['Model'], categories=["vanilla", "cot"], ordered=True)

    # Sort the DataFrame by 'Model Class' and then by 'Model'
    df = df.sort_values(by=['Model Class'])#, 'Model'])

    # Convert the DataFrame to LaTeX format with multi-row structure and horizontal lines
    latex_table = df.to_latex(index=False, column_format='|l|l|c|c|', header=['Model Class', 'Model', 'Event Likelihood', 'Speed Inferences'], escape=False, multicolumn=True)

    # Add horizontal lines to separate model classes
    latex_table_lines = latex_table.splitlines()
    new_latex_table = []
    last_class = None
    for line in latex_table_lines:
        if 'Model Class' in line or '\\hline' in line:
            new_latex_table.append(line)
        else:
            current_class = line.split('&')[0].strip()
            if current_class != last_class:
                if last_class is not None:
                    new_latex_table.append('\\hline')
                last_class = current_class
            new_latex_table.append(line)

    new_latex_table = '\n'.join(new_latex_table)

    # Output the LaTeX table
    print(new_latex_table)
    return new_latex_table


def compare_res(res_df, human_agg_res, gt_human_agg_res, stim2baselines, sorted_stimuli_ids): 
    comp_human = []
    comp_model = [] 
    comp_model_worst = []
    comp_model_agg = []
    comp_human_se = []
    comp_model_opt = []

    run_baselines= True
    if run_baselines: 
        baseline_model_tags = stim2baselines[sorted_stimuli_ids[0] + ".txt"].keys()
        comp_baselines = {model_tag: [] for model_tag in baseline_model_tags}

    question_types = []

    diff_stim = []
    check_stim =[]
    close_stim=[]

    comp_gt = False

    stim2model = {stim_id: {} for stim_id in sorted_stimuli_ids}

    for stim_id in sorted_stimuli_ids:
        stim_id = stim_id + ".txt" 

        # if "team-consistent-short-variant" in stim_id: continue
        if "all" not in human_agg_res[stim_id]: continue 
        resps = human_agg_res[stim_id]["all"]
        # if len(resps) < 5: continue
        
        tag, question = stim_id.split("*")
        question =question.split(".txt")[0]
        
        model_res = res_df.loc[res_df.stimuli_id == stim_id]
        if len(model_res) == 0: 
            print("skipping ", stim_id, " no model res")
            continue 
        model_res = model_res.iloc[0]
        model_scores = model_res["model_scores"] 
        model_means = model_res["model_means"]
        sampled_model_means = model_res["new_means"]
        
        human_agg = human_agg_res[stim_id]["agg"]
        gt_human_agg = gt_human_agg_res[stim_id]["agg"]
        human_se = human_agg_res[stim_id]["se"]
        
        
        if len(model_scores) == 0 or  model_means == 'null':
            print("ERROR SCORES = 0", model_scores, model_means, stim_id)
            continue 
        
        if len(model_scores) != len(model_means): 
            print("ERROR: ", stim_id, model_scores, model_means, human_agg)
            continue
        best_model = np.argmax(model_scores)
        worst_model = np.argmin(model_scores)
        
        best_model_score = model_means[best_model]
        worst_model_score = model_means[worst_model]
        agg_model_score = np.mean([m * s for m, s in zip(model_means, model_scores)])
        
        
        if "fast" in question: 
            check_stim.append([stim_id, model_means, model_scores,human_agg])
            
            # if stim_id == "relay-race-5*False*How_fast_is_Ernie?.txt": 
            #     # patch polarity for now
            #     if best_model_score > 50: best_model_score = 100 - best_model_score
            #     if worst_model_score > 50: worst_model_score = 100 - worst_model_score
            
            question_types.append("how_fast")

        else: 
            question_types.append("how_likely")
            
        if comp_gt:
            comp_human.append(gt_human_agg)
        else: comp_human.append(human_agg) 
        
        comp_human_se.append(human_se)

        # TODO -- bad code
        # print("new means: ", sampled_model_means, best_model, compute_se(sampled_model_means[best_model]))
        best_se = compute_se(sampled_model_means[best_model])
        comp_model.append([best_model_score, best_se])
        
        worst_se = compute_se(sampled_model_means[worst_model])
        comp_model_worst.append([worst_model_score, worst_se])
        comp_model_agg.append(agg_model_score)
        if run_baselines: 
            for model_tag in baseline_model_tags: 
                comp_baselines[model_tag].append(stim2baselines[stim_id][model_tag])

        # opt
        if comp_gt: 
            h_compare = gt_human_agg
        else: h_compare = human_agg

        h_dists = [np.abs(h_compare - x) for x in model_means]
        min_dist_idx = np.argmin(h_dists)
        min_dist_mean = model_means[min_dist_idx]
        comp_model_opt.append(min_dist_mean)

        stim2model[stim_id] = {"best_model_idx": best_model, "worst_model_idx": worst_model, "model_scores": model_scores, "model_means": model_means, "opt_model_idx": min_dist_idx}
        
        if np.abs(human_agg - best_model_score) > 30 or np.abs(min_dist_mean - best_model_score) > 20: 
            print("DIFFERENT: ", stim_id.split(".txt")[0], "\n\tHuman agg: ", human_agg,f"\n\t Opt: {min_dist_mean} ({min_dist_idx}), Best score: {best_model_score} ({best_model}) Worst score: {worst_model_score} ({worst_model})\n\t", model_scores)
                #"\n\t", best_model, worst_model, model_scores)
            diff_stim.append((stim_id, human_agg, best_model_score, worst_model_score))
        
        if np.abs(human_agg - best_model_score) < 5: 
            # print(stim_id, human_agg, best_model_score, worst_model_score, best_model, worst_model, model_scores)
            close_stim.append((stim_id, human_agg,  best_model_score, worst_model_score, best_model, worst_model, model_scores))   

    return comp_model, comp_model_worst, comp_human, comp_human_se, comp_baselines, question_types
        
        
        


def per_query_plots(stim, all_stimuli_resps, srcs, n_models_keep=25, 
                    n_queries=8, src2viz={}, density=False, n_bins=25, 
                    save_dir="./figs/", show_plots=False, 
                    colors=['red', 'blue', 'green', 'purple', 'grey', 'pink', 'orange', 'black'], 
                    both_model_types=False,show_title=False): 
    

    M = len(srcs)

    fig = plt.figure(figsize=(18, 8 * M))
    outer_grid = GridSpec(3, 3, wspace=0.1, hspace=0.1)
    # fig = plt.figure(figsize=(26,  4 * M))
    # outer_grid = GridSpec(2, 4, wspace=0.4, hspace=0.1)

    bin_range = [-5, 105]
    
    if src2viz == {}: src2viz = {src: src for src in srcs}

    for i in range(n_queries): 
        check_query = f"query{i+1}"
        inner_grid = outer_grid[i].subgridspec(M, 1, hspace=0.1)

        for j, src in enumerate(srcs):
            ax = fig.add_subplot(inner_grid[j])
            color = colors[j]
            
            if both_model_types: 
                # both explicit and implicit 
                base_stim = "_".join(stim.split("_")[:-1])
                for m_idx, model_type in enumerate(["explicit", "implicit"]): 
                
                    stim = f"{base_stim}_{model_type}"
                    try: resps = all_stimuli_resps[stim][src]
                    except: resps = []
                    
                    # print(src, len(resps), model_type)
                    
                    ax.hist([s[check_query] for s in resps], label=f"{src2viz[src]} ({model_type.capitalize()})", density=density, alpha=0.5, bins=n_bins, range=bin_range, color=color if m_idx == 0 else "grey")
            else: 
                # just show the stimuli requested
                try: 
                    resps = all_stimuli_resps[stim][src]
                    if resps is None: resps=  []
                except: 
                    resps = []
                
                # print(src, len(resps))
            
                ax.hist([s[check_query] for s in resps], label=src2viz[src], density=density, alpha=0.5, bins=n_bins, range=bin_range, color=color)
            ax.legend(loc='upper right')
            ax.set_xlim(bin_range)
            if density != True: 
                ax.set_ylim([0, n_models_keep+1])
            if j == 0:
                ax.set_title(f"Query {i+1}", fontsize=18)
                
            if j != M-1: ax.set_xticks([])

    fpth = save_dir + f"query_hists_{stim}.png"
    if show_title: plt.suptitle(stim, fontsize=18)
    plt.savefig(fpth)
    
    if not show_plots: 
        plt.close()
    
    return fpth

import numpy as np
from scipy.stats import pearsonr, norm

def pearsonr_ci(x, y, alpha=0.05):
    # help from GPT -- check!!
    # Calculate Pearson correlation
    r, pval = pearsonr(x, y)

    # Fisher z-transformation
    z = np.arctanh(r)
    
    # Standard error of z
    se = 1 / np.sqrt(len(x) - 3)
    
    # Calculate the z critical value for 95% CI
    z_critical = norm.ppf(1 - alpha / 2)
    
    # Calculate the CI in the z-space
    z_ci_lower = z - z_critical * se
    z_ci_upper = z + z_critical * se
    
    # Transform the CI back to the r-space
    r_ci_lower = np.tanh(z_ci_lower)
    r_ci_upper = np.tanh(z_ci_upper)
    
    return r, pval, (r_ci_lower, r_ci_upper)



def query_scatterplot(resps, 
                      x_queries = [], 
                      y_queries = [],
                      c_queries = [], 
                      x_label = "Effort (P1 - P2)", 
                      y_label = "Skill (P1 - P2)",
                      c_label = None,
                      alpha = 0.5,
                      jitter_dev = 1.0,
                      title=None,
                      variant = "implicit",
                      file_tag = "query_scatter.png"): 
    
    if len(x_queries) == 1: 
        x_vals = [r[x_queries[0]] for r in resps]
    else: 
        x_vals = [r[x_queries[0]] - r[x_queries[1]] for r in resps]
        
    if len(y_queries) == 1: 
        y_vals = [r[y_queries[0]] for r in resps]
    else: 
        y_vals = [r[y_queries[0]] - r[y_queries[1]] for r in resps]
        
    if len(c_queries) == 1: 
        c_vals = [r[c_queries[0]] for r in resps]
    else: 
        c_vals = [r[c_queries[0]] - r[c_queries[1]] for r in resps]
    

    fig, ax = plt.subplots(figsize=(8, 6))#3,1, figsize=(6,14))

    # print(analysis_utils.pearsonr_ci(x_vals, y_vals))

    x_vals = [np.random.normal(0, jitter_dev) + x for x in x_vals]
    y_vals = [np.random.normal(0, jitter_dev) + x for x in y_vals]

    scatter= ax.scatter(x_vals, y_vals, alpha=alpha, 
                        #label=sport, 
                        s=100, c=c_vals, cmap='viridis_r', vmin=0, vmax=100)   
    cb = plt.colorbar(scatter) 
    cb.set_label(label=c_label, fontsize=18,) #weight='bold')
    # ax.legend()
    ax.set_xlim([-110, 110])
    ax.set_ylim([-110, 110])
    ax.set_ylabel(y_label, fontsize = 18)
    ax.set_xlabel(x_label, fontsize = 18)

    # Adding dashed quadrant lines
    ax.axhline(0, color='black', linestyle='--', lw=1)
    ax.axvline(0, color='black', linestyle='--', lw=1)

    if title is not None:
        ax.set_title(title, fontsize=18)
        
    plt.tight_layout()
    plt.savefig(file_tag, dpi=400)


def per_query_plots_diff_sports(base_stim, all_resps, srcs, n_models_keep=25, 
                    n_queries=8, src2viz={}, density=False, n_bins=25, 
                    save_dir="./figs/", show_plots=False, 
                    colors=['red', 'blue', 'green', 'purple', 'grey', 'pink', 'orange', 'black'], 
                    both_model_types=False,show_title=False): 
    
    
    # TODO --- refactor --- adjusted to make separate plots per sport
    # srcs here are sports
    

    M = len(srcs)

    fig = plt.figure(figsize=(18, 8 * M))
    outer_grid = GridSpec(3, 3, wspace=0.1, hspace=0.1)
    # fig = plt.figure(figsize=(26,  4 * M))
    # outer_grid = GridSpec(2, 4, wspace=0.4, hspace=0.1)

    bin_range = [-5, 105]
    
    if src2viz == {}: src2viz = {src: src for src in srcs}

    for i in range(n_queries): 
        check_query = f"query{i+1}"
        inner_grid = outer_grid[i].subgridspec(M, 1, hspace=0.1)

        for j, src in enumerate(srcs):
            ax = fig.add_subplot(inner_grid[j])
            color = colors[j]
            
            if both_model_types: 
                # both explicit and implicit 
                #base_stim = "_".join(stim.split("_")[:-1])
                for m_idx, model_type in enumerate(["explicit", "implicit"]): 
                #for m_idx, model_type in enumerate(["implicit", "explicit",]): 
                    stim = f"{src}_{base_stim}_{model_type}"
                    try: resps = all_resps[stim]
                    except Exception as e:
                        resps = []
                    
                    # print(src, len(resps), model_type)
                    
                    ax.hist([s[check_query] for s in resps], label=f"{src2viz[src]} ({model_type.capitalize()})", density=density, alpha=0.5, bins=n_bins, range=bin_range, color=color if m_idx == 0 else "grey")
            else: 
                # just show the stimuli requested
                try: 
                    stim = f"{src}_{base_stim}_{model_type}"
                    resps = all_resps[stim]#][src]
                    if resps is None: resps=  []
                except Exception as e:
                    resps = []
                
                # print(src, len(resps))
            
                ax.hist([s[check_query] for s in resps], label=src2viz[src], density=density, alpha=0.5, bins=n_bins, range=bin_range, color=color)
            ax.legend(loc='upper right')
            ax.set_xlim(bin_range)
            if density != True: 
                ax.set_ylim([0, n_models_keep+1])
            if j == 0:
                ax.set_title(f"Query {i+1}", fontsize=18)
                
            if j != M-1: ax.set_xticks([])

    fpth = save_dir + f"query_hists_{stim}.png"
    if show_title: plt.suptitle(stim, fontsize=18)
    plt.savefig(fpth)
    
    if not show_plots: 
        plt.close()
    
    return fpth






    
def pair_plots(stim, all_stimuli_resps, srcs, queries=[1,2,4], query_map={}, src2viz={}, save_dir="figs/", show_plots=False): 

    fpths= []
    for src in srcs: 
        
        resps = all_stimuli_resps[stim][src]
        print(resps)
        fig, ax = plt.subplots()
        q1, q2, q3 = queries

        x1s = [] 
        x2s = [] 
        x3s = []
        for entry in resps: 
            if type(entry) == list: 
                # format is index then (0-7) [make this cleaner!]
                
                # some people gave diff click lists -- skip for now diff len b/c breaks coloring
                clicks = [entry[q1-1], entry[q2-1], entry[q3-1]]
                if not all(len(arr) == len(clicks[0]) for arr in clicks):
                    continue
                
                x1s.extend(entry[q1-1])
                x2s.extend(entry[q2-1])
                x3s.extend(entry[q3-1])
                
                # x1s.append(np.mean(entry[q1-1]))
                # x2s.append(np.mean(entry[q2-1]))
                # x3s.append(np.mean(entry[q3-1]))
            else: 
                x1s.append(entry[f"query{q1}"])
                x2s.append(entry[f"query{q2}"])
                x3s.append(entry[f"query{q3}"])
            
        jitter_level = 0.25
        x1s = [x + np.random.normal(0,jitter_level) for x in x1s]
        x2s = [x + np.random.normal(0,jitter_level) for x in x2s]
        x3s = [x + np.random.normal(0,jitter_level) for x in x3s]
            
            
        x3s = np.array(x3s)
        print(x3s)
            
        alpha=0.7
        size=100

        plt.scatter(x1s, x2s, c=x3s, cmap='viridis', s=size, alpha=alpha,vmin=0, vmax=100)
        cbar = plt.colorbar()
        cbar.set_label(f'{query_map[q3]}')

        min_scale = -5 
        max_scale = 105
        plt.plot([min_scale, max_scale], [min_scale, max_scale], color='grey', linestyle='--', label='y = x')

        plt.xlim([min_scale, max_scale])
        plt.ylim([min_scale, max_scale])
        
        if src2viz is not None:  plt.title(src2viz[src], fontsize=20)
        else: plt.title(src.capitalize(), fontsize=20)
        plt.xlabel(query_map[q1], fontsize=18)
        plt.ylabel(query_map[q2], fontsize=18)
        
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        fpth = save_dir + f"query_pairs_{src}_{q3}.png"
        fpths.append(fpth)
        plt.savefig(fpth)
        if not show_plots: plt.close()

    return fpths
    
                
                
                
def aggregate_plots(sorted_stimuli_ids, our_res, human_res, llm_res, question_types, model2viz, question_type2viz, 
                    prompt_methods=["cot", "vanilla"], sim_noise=2.5, use_sim_humans=False, with_llms=True, ylim=[0,35], n_queries=8,
                    normalize=False): 


    model_types = ["ours"] + [f"llm_{method}" for method in prompt_methods]
    combo_res = {}
    for question_type in question_types.keys(): 
        combo_res[question_type] = {"human": []}
        for model in model_types: 
            combo_res[question_type][model] = []


    for stim in sorted_stimuli_ids: 
        
        #if "diving" not in stim: continue
        #if "implicit" not in stim: continue
        
        if len(human_res[stim]) == 0 or np.any([llm_res[prompt_method][stim] is None for prompt_method in prompt_methods]) or len(our_res[stim]["agg_samp_means"]) == 0: #llm_res["cot"][stim] is None or llm_res["vanilla"][stim] is None: 
            print("skipping: ", stim, len(human_res[stim]), len(our_res[stim]["agg_samp_means"]),  [llm_res[prompt_method][stim] is None for prompt_method in prompt_methods])
            continue

        samp_resps = our_res[stim]["agg_samp_means"]
        
        # TODO -- move elswhere
        # filter malformed queries
        samp_resps = [s for s in samp_resps if np.all(["query" in k for k in s.keys()])]

        if use_sim_humans: 
            h_split, m_split = np.array_split(samp_resps, 2)
        else: 
            m_split = samp_resps
            h_split = human_res[stim]

        if with_llms:
            try: 
                llm_resps = {method: llm_res[method][stim] for method in prompt_methods}
            except: 
                llm_resps = {method: [{f"query{idx+1}": -1 for idx in range(n_queries)}] for method in prompt_methods}
        
        for question_type, idxs in question_types.items(): 
            for idx in idxs:
                
                if idx in {}: continue 
                
                query = f"query{idx}"

                combo_res[question_type]["human"].append([s[query] for s in h_split])
                combo_res[question_type]["ours"].append([s[query] for s in m_split])
                if with_llms:
                    for method in prompt_methods:
                        combo_res[question_type][f"llm_{method}"].append([s[query] for s, _ in llm_resps[method]])
        
        
    for model_type in model_types:
        #print(model_type)
        if not with_llms and "llm" in model_type: continue
        # fig, axes = plt.subplots(1,len(question_types),figsize=(18,6))
        # fig2, axes2 = plt.subplots(1,len(question_types),figsize=(18,6))
        fig, axes = plt.subplots(1,len(question_types),figsize=(14,4))
        fig2, axes2 = plt.subplots(1,len(question_types),figsize=(14,4))

        show_err = True 
        for i, question_type in enumerate(question_types.keys()):
            ax = axes[i]
            ax2 = axes2[i]

            human_sub = combo_res[question_type]["human"]
            h_means = [np.mean(x) for x in human_sub]
            if use_sim_humans and sim_noise is not None: 
                h_means = [x + np.random.normal(0, sim_noise) for x in h_means]
            h_se = [compute_se(x) for x in human_sub]
            
            
            model_sub = combo_res[question_type][model_type]
            m_means = [np.mean(x) for x in model_sub]
            
            
            
            m_se = [compute_se(x) for x in human_sub]
            
            # if model_type == "ours": print(m_means)
            #print(len(m_means), len(h_means))
            if len(m_means) == 0 or len(h_means) == 0: continue
            corr = pearsonr(m_means, h_means)[0]
            
            if normalize: 
            
                from scipy.optimize import curve_fit

                def linear_function(x, a, b):
                    return a * x + b

                # # Assuming h_means and m_means are your data
                # h_means = [np.mean(x) for x in h]
                # m_means = [np.mean(x) for x in m]

                # Perform the curve fitting
                popt, _ = curve_fit(linear_function, m_means, h_means)

                # Extract the optimal parameters
                scaling_factor = popt[0]
                offset = popt[1]
                print("scaling factor: ", scaling_factor, " offset: ", offset)

                m_means = scaling_factor * np.array(m_means) + offset
                
                corr = pearsonr(m_means, h_means)[0]          
            
            
            
            label = f"{model2viz[model_type]} (r = {round(corr,2)})" 
            #label += f", r_post = {round(corr_post,2)})"
            
            
            sns.regplot(x=m_means, y=h_means, ax=ax, 
                        #color=color, 
                        scatter=True,
                                label=label, ci=95)
            
            print(len(m_means), len(h_means))
            
            if show_err: 
                ax.errorbar(m_means, h_means,
                    yerr = h_se,
                    xerr = m_se, 
                    #color=color,
                    alpha=0.3,
                        #fmt ='o',
                    fmt='none', capsize=5, zorder=1) #color='C0')
            
            
                
            lims = [
                -5, 105
            ]

            # now plot both limits against eachother
            ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
            ax.legend(fontsize=12)


            ax.set_xlim(lims)
            ax.set_ylim(lims)

            ax.set_xlabel("Model Prediction", fontsize=18)
            if i == 0: 
                if use_sim_humans: 
                    ax.set_ylabel("Sim-Human Prediction", fontsize=18)
                
                else: 
                    ax.set_ylabel("Human Prediction", fontsize=18)
                ax2.set_ylabel("Counts", fontsize=18)
            ax.set_title(question_type2viz[question_type], fontsize=18)

            # if model_type == "ours" and use_sim_humans: 
            #     all_means = np.concatenate([h_means, m_means])
            # else: all_means = m_means 
            # print(np.shape(all_means))
            ax2.hist(m_means, alpha=0.5, density=False, label="Model")
            ax2.hist(h_means, alpha=0.5, density=False, label="Human")
            
            ax2.set_xlim(lims)
            ax2.set_ylim(ylim)
            ax2.set_xlabel("Response",fontsize=18)
            
            ax2.legend()
            
    return combo_res


from PIL import Image
import os

import ast
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_scenario(self, scenario_text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 10, scenario_text.encode('latin-1', 'replace').decode('latin-1'))
    
    def add_plot(self, image_path, x=10, y=50, w=150):
        self.image(image_path, x=x, y=y, w=w)

def pngs_to_pdf(image_directory='participant_resps/', output_pth="participant_hists.pdf", remove_orig=True, target_width=None, min_height=None):
    # convert a directory of pngs to PDF
    
    if os.path.exists(output_pth):
        os.remove(output_pth)

    # Get a list of all PNG files in the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.png')]

    # Sort the files to ensure they are in the correct order
    image_files.sort()

    # Load the images
    images = [Image.open(os.path.join(image_directory, f)) for f in image_files]

    # Convert the images to RGB (if they are not already)
    images_rgb = [img.convert('RGB') for img in images]

    # Resize the second image if target width is provided
    if target_width and len(images_rgb) > 1:
        second_img = images_rgb[1]
        width_percent = (target_width / float(second_img.size[0]))
        target_height = int((float(second_img.size[1]) * float(width_percent)))
        
        # Adjust height if it's smaller than the minimum height
        if min_height and target_height < min_height:
            target_height = min_height
            target_width = int(float(min_height) * (second_img.size[0] / float(second_img.size[1])))

        images_rgb[1] = second_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Save the images as a PDF
    images_rgb[0].save(output_pth, save_all=True, append_images=images_rgb[1:])

    if remove_orig:
        for f in image_files:
            os.remove(os.path.join(image_directory, f))
            
def count_occurrences_with_whitespace(substring, main_string):
    # help from ChatGPT
    # Create a pattern to match any leading whitespace before the substring
    pattern = rf'\s+{re.escape(substring.strip())}'
    
    # Find all occurrences of the pattern in the main string
    matches = re.findall(pattern, main_string)
    
    return len(matches)
            
def assess_function_use(world_model):
    ''' 
    Check if any functions are defined but not used
    TODO: this should be checked carefully; make sure no loopholes
    ''' 
    full_model = world_model.to_string()
    # don't look at the conditions
    #model_body = full_model.split("// BACKGROUND KNOWLEDGE")[-1].split("// CONDITIONS")[0]
    defns = [entry[0] for entry in world_model.definitions]
    unused = []
    for defn in defns: 
        occs = count_occurrences_with_whitespace(defn, full_model)
        print(defn, occs) 
        # each function should be defined and used with at least one condition
        # so anything < 3 isn't used (NOTE: check condition count though -- todo)
        if occs < 3:  
            unused.append(defn)
    return unused
    
def get_baseline_inference(world_model): 
    # NOTE -- this is just for the commentary case
    # and assumes that commentary is in the last condition
    
    # ablate the last condition and re-run inference
    
    return 


def process_samples(samples, burn_in=500, subsample=-1, n_queries=8): 
    # process the data that we have 
    chain_samples = eval(samples)[burn_in:]
    
    if subsample != -1: 
        chain_samples = np.random.choice(chain_samples, subsample)
    
    
    query_tags = set(chain_samples[0]["value"].keys()) # all queries are computed per sample 
    if len(query_tags) != n_queries: 
        print("fewer queries: ", query_tags)
        return None
    
    
    query_samples = {query: [] for query in query_tags}
    query_aggs = {}
        
    for entry in chain_samples: 
        for query, val in  entry["value"].items():
            query_samples[query].append(val)
    
    
    #if 'Query "1"' in query_tags: # correct some odd fmt 
    if "query1" not in query_tags: 
        print("Fixing parse!", query_tags)
        new_query_samples = {f"query{idx+1}": query_samples[f'Query "{idx+1}"'] for idx in range(n_queries)}#query_tags = 
        query_samples = new_query_samples
    
    query_aggs = {query: np.mean(samps) for query, samps in query_samples.items()}
    query_aggs = dict(sorted(query_aggs.items()))
    # c has all samples per query
    return query_aggs, query_samples
    
    
            
def process_single_rollout(rollout_idx, stim_dir, ran_commentary=False, burn_in=500, 
                           subsample=-1, n_queries=8,
                           inference_method="MCMC", n_samples=5000,
                           timeout=120): 
    # NOTE --- this needs to be cleaned up!!!! 
    # NOTE: inference_method and n_samples need to be set to what was run
    
    res = {"all_raw_samples": None, 
           "defn_count": None,
           "world_model": None,
           "main_agg_queries": None, 
           "main_query_samples": None, 
           "unused_funcs": None, 
           "base_agg_queries": None,
           "base_query_samples": None}

    try:
        #with open(f"{stim_dir}llm-ppl_rollout{rollout_idx}.json", "r") as f: 
        with open(f"{stim_dir}/inference_results_{rollout_idx}.json", "r") as f: 
            res = json.load(f)
            model_str = res['wm']["model"]
            res = res['wm']["model_metadata"]
            executable = res["executability"]
            samples = res["official_posterior_samples"]
            res["all_raw_samples"] = samples

            world_model = model_utils.WorldModel(model_str)
            n_defns = len(world_model.definitions)
            res["defn_count"] = n_defns
            res["world_model"] = world_model
            
            unused_funcs = assess_function_use(world_model)
            res['unused_funcs'] = unused_funcs

            # process the data that we have 
            print("initial processing")
            resp = process_samples(samples, burn_in, subsample, n_queries)
            print("processed")
            if resp is None: return None
            else: query_aggs, processed_samps = resp
            
            res["main_agg_queries"] = query_aggs
            res["main_query_samples"] = processed_samps
            
            if ran_commentary:
                
                samples = run_ablated_commentary(rollout_idx, stim_dir, 
                           inference_method, n_samples,
                           timeout)
                
            
                
                # # if we're in the commentary case, rerun inference [and reprocess]
                # # TODO: make this nicer
                # # TODO -- update to_string to use the same sampling method...
                # print("ablated")
                # model_str = world_model.to_string(posterior_samples=n_samples, 
                #                                     sampling_method=inference_method)
                # model_str_lines = model_str.splitlines()
                # lines_with_condition = [(i, line) for i, line in enumerate(model_str_lines) if "condition(" in line]
                # # ablate last
                # commentary_i, commentary_str = lines_with_condition[-1]
                # model_str_lines[commentary_i] = "//" + commentary_str
                # # also check if any other lines don't start with "In the ... match," -- then ablate them
                # # NOTE -- we should make this more rigorous later to cross-check that we're not missing other multi-line 
                # for commentary_i, commentary_str in lines_with_condition[:-1]: 
                #     print("commentary str: ", commentary_str)
                    
                #     # get comment before
                #     # NOTE -- this parsing assumes that each line has the comment directly above
                #     # NOTE -- this may not work otherwise -- we should cross-check!!!
                #     commentary_comment = model_str_lines[commentary_i - 1]
                #     if "//" in commentary_comment: 
                #         if "In the" not in commentary_comment: 
                #             model_str_lines[commentary_i] = "//" + commentary_str
                #     else: 
                #         print("!!!!WARNING: Comment not included before condition")
                #     # if "In the" not in commentary_str: 
                #     #     model_str_lines[commentary_i] = "//" + commentary_str
                
                # model_str = "\n".join(model_str_lines)
                # print("sampling")
                # print("model: ", model_str)
                # keys, samples = utils.run_webppl(model_str, timeout=timeout)
                resp = process_samples(samples, burn_in, subsample, n_queries)
                print("output")
                if resp is None: return None
                else: query_aggs, processed_samps = resp
                res["base_agg_queries"] = query_aggs
                res["base_query_samples"] = processed_samps
 
            return res 

    except Exception as e:
        print("Error in processing: ", e)
        return None

def get_transform_who_would_win_results(who_would_win_baseline_samples, baseline_query='query8', take_negative=True):
    if type(who_would_win_baseline_samples) == str: who_would_win_baseline_samples = eval(who_would_win_baseline_samples)
    # Transformation: 
    baseline_samples = np.array([s['value'][baseline_query] for s in who_would_win_baseline_samples])
    if take_negative: 
        baseline_samples = -baseline_samples # Flip the sign since team1 winning is low on the human data scale. 

    from sklearn.preprocessing import QuantileTransformer
    scaler = QuantileTransformer()
    scaler.fit(baseline_samples.reshape(-1, 1))
    return scaler

def transform_who_would_win_results(initial_values, who_would_win_transformer, take_negative=True, scale=100):
    transformed_values = np.array(initial_values)
    if take_negative:
        transformed_values = -transformed_values
    
    transformed_values = who_would_win_transformer.transform(transformed_values.reshape(-1, 1))
    transformed_values = transformed_values * scale
    transformed_values = list(transformed_values.flatten())
    return transformed_values


    # transform_min, transform_max, transform_std = who_would_win_transformer
    # transformed_values = []
    # # This transformation is too severe and is quashing the 
    # print(transform_min, transform_max)
    # for v in initial_values:
    #     v = -v  # Flip the sign since team1 winning is low on the human data scale.
    #     # Preserve symmetries.
    #     if v <= 0:
    #         curr_min = transform_min
    #         curr_max = 0

    #         # Transform from -1 to 0
    #         b, a = -1, 0
    #     else:  
    #         curr_min = 0    
    #         curr_max = transform_max
    #         # Transform from 0 to 1
    #         b, a = 0, 1
    #     transformed_v = (v - curr_min)*(b-a) / (curr_max - curr_min)
    #     transformed_v = transformed_v * v_scaler
    #     transformed_v += x_shift # Since 0 is centered around 50 in the human data.
    #     transformed_values.append(transformed_v)
    # return transformed_values

   

    
def get_who_would_win_and_how_much_samples(rollout_idx, stim_dir, 
                           inference_method="MCMC", n_samples=20000,
                           timeout=240,
                           override=False,
                           dont_recompute=False,
                           maximum_queries=8,
                           
                           query_to_keep=8,
                           file_tag='inference_results'):
    samples = None
    output_file = f"{stim_dir}/who_would_win_and_by_how_much_results_{rollout_idx}.json"
    if not override and os.path.exists(output_file): 
        with open(output_file, "r") as f:
            print("loading who would win and how much samples from file!") 
            samples = json.load(f)
            return samples['ablated_samples']
    
    if dont_recompute:
        print(f"Who would win and how much examples not found for: {stim_dir}/{file_tag}_{rollout_idx}.json, continuing")
        return None

    # otherwise --- compute who would win and how much and save out
    print(f"computing who would win and how much samples for: {stim_dir}/{file_tag}_{rollout_idx}.json")
    with open(f"{stim_dir}/{file_tag}_{rollout_idx}.json", "r") as f: 
        res = json.load(f)
        model_str = res['wm']["model"]
        res = res['wm']["model_metadata"]
        samples = res["official_posterior_samples"]
        res["all_raw_samples"] = samples

        world_model = model_utils.WorldModel(model_str)

        # Just directly remove all conditions.
        model_str_lines = model_str.splitlines()
        lines_with_condition = [(i, line) for i, line in enumerate(model_str_lines) if "condition(" in line]
        for commentary_i, commentary_str in lines_with_condition:
            model_str_lines[commentary_i] = "//" + commentary_str
        # Also directly remove all queries.
        for i, line in enumerate(model_str_lines):
            if "query" in line:
                if f"query{query_to_keep}" in line:
                    pass
                else:
                    for j in range(1, maximum_queries):
                        if f"query{j}" in line:
                            model_str_lines[i] = "//" + line
        model_str = "\n".join(model_str_lines)
        print("sampling")
        print("model: ", model_str)
        keys, samples = utils.run_webppl(model_str, timeout=timeout)
        with open(output_file, "w") as f: 
            
            save_data = {"ablated_model": model_str, 
                         "ablated_samples": samples}
            json.dump(save_data, f)
            
        return samples
        
    return samples

    
def rmse(x1, x2): 
    squared_diff_sum = sum((a - b) ** 2 for a, b in zip(x1, x2))
    return (squared_diff_sum / len(x1)) ** 0.5 
    
def get_ablated_commentary_samples(rollout_idx, stim_dir, 
                           inference_method="MCMC", n_samples=10000,
                           timeout=240,
                           override=False,
                           dont_recompute=False): 
    
    samples = None
    ablated_save_file = f"{stim_dir}/ablated_results_{rollout_idx}.json"
    if not override and os.path.exists(ablated_save_file): 
        with open(ablated_save_file, "r") as f:
            print("loading ablated samples from file!") 
            samples = json.load(f)
            return samples['ablated_samples']
    
    if dont_recompute:
        print(f"Ablated examples not found for: {stim_dir}/inference_results_{rollout_idx}.json, continuing")
        return None

    # otherwise --- compute ablated and save out
    print(f"computing ablated samples for: {stim_dir}/inference_results_{rollout_idx}.json")
    with open(f"{stim_dir}/inference_results_{rollout_idx}.json", "r") as f: 
        res = json.load(f)
        model_str = res['wm']["model"]
        res = res['wm']["model_metadata"]
        executable = res["executability"]
        samples = res["official_posterior_samples"]
        res["all_raw_samples"] = samples

        world_model = model_utils.WorldModel(model_str)
        # if we're in the commentary case, rerun inference [and reprocess]
        # TODO: make this nicer
        # TODO -- update to_string to use the same sampling method...
        print("ablated")
        model_str = world_model.to_string(posterior_samples=n_samples, 
                                            sampling_method=inference_method)
        model_str_lines = model_str.splitlines()
        lines_with_condition = [(i, line) for i, line in enumerate(model_str_lines) if "condition(" in line]
        # ablate last
        commentary_i, commentary_str = lines_with_condition[-1]
        model_str_lines[commentary_i] = "//" + commentary_str
        # also check if any other lines don't start with "In the ... match," -- then ablate them
        # NOTE -- we should make this more rigorous later to cross-check that we're not missing other multi-line 
        for commentary_i, commentary_str in lines_with_condition[:-1]: 
            #print("commentary str: ", commentary_str)
            
            # get comment before
            # NOTE -- this parsing assumes that each line has the comment directly above
            # NOTE -- this may not work otherwise -- we should cross-check!!!
            commentary_comment = model_str_lines[commentary_i - 1]
            if "//" in commentary_comment: 
                if "In the" not in commentary_comment: 
                    model_str_lines[commentary_i] = "//" + commentary_str
            else: 
                print("!!!!WARNING: Comment not included before condition")
            # if "In the" not in commentary_str: 
            #     model_str_lines[commentary_i] = "//" + commentary_str
        
        model_str = "\n".join(model_str_lines)
        #print("sampling")
        #print("model: ", model_str)
        keys, samples = utils.run_webppl(model_str, timeout=timeout)
        with open(ablated_save_file, "w") as f: 
            
            save_data = {"ablated_model": model_str, 
                         "ablated_samples": samples}
            json.dump(save_data, f)
            
        return samples
        
    return samples


    

def run_agg_analyses(experiment_dir, llm_only=False): 
    ''' 
    Get aggregate stats and initial analyses on the saved data for a particular scenario
    Note: we may want different analyses for the LLM only cases (TODO)
    
    For model-synthesis:
    - compile_rate 
    - TODO: other stats on the quality of the generated models (e.g., whether functions are called at least once)
    '''
    
    if llm_only: return # TODO 
    
    analysis_dir = f"{experiment_dir}/analyses/"
    print(f"\n====Exporting analyses to  ====\n{analysis_dir}\n\n")

    if not os.path.exists(analysis_dir): os.makedirs(analysis_dir)
    
    # parsing out the stimuli name
    # NOTE -- this is based on our particular format (Meta-model and effort)
    # should make this less brittle going forwards
    scenario_name = experiment_dir.split("_Meta")[0].split("effort_")[1]
    ran_commentary = False
    if "-likely" in scenario_name:
        ran_commentary = True
        # then this question has to do with commentary... [again, brittle!]
        tag = scenario_name.split("implicit_continuous_variable_")[1].split("-likely")[0]
        trgt_question, directionality = tag.split("_")
        if trgt_question == "new-1": trgt_q_idx = 7
        else: trgt_q_idx = 8

    # get compile rate 
    # the file format is "generated_model_{run-idx}_{success}.txt"
    print([file.split("_") for file in os.listdir(f"{experiment_dir}/compiled_models/")])
    success_runs = [file.split("_")[2] for file in os.listdir(f"{experiment_dir}/compiled_models/")]
    failed_runs = [file.split("_")[2] for file in os.listdir(f"{experiment_dir}/failed_models/")]
    
    n_success = len(success_runs)
    tot_runs = n_success + len(failed_runs)
        
    # load in the data from the success cases and show some visualization for them
    other_parse_failures = []
    agg_res = []
    success_res = {}
    for rollout_idx in success_runs: 
        res = process_single_rollout(rollout_idx, experiment_dir, ran_commentary)
        if res is None or "main_agg_queries" not in res: 
            other_parse_failures.append(rollout_idx)
            continue # note -- these need to be factored into compile rate!!
        else: 
            # success! 
            # TODO -- make a nice visualization
            success_res[rollout_idx] = res
            agg_res.append([rollout_idx, res['main_agg_queries']])
        
       
    n_other_fail = len(other_parse_failures) 
    with open(f"{analysis_dir}/agg_stats.txt", "w") as f: 
        f.write(f"Compile rate: {n_success} out of {tot_runs} ({round(n_success / tot_runs * 100, 2)}%)")
        if n_other_fail != 0: 
            f.write(f"\nAn additional {len(other_parse_failures)} failed due to an error in proposed query format")
            
        if len(agg_res) != 0:
            f.write("\n\nAvg responses for each query (of successful models)")
        for rollout_idx, query_res in agg_res: 
            
            rollout_res = success_res[rollout_idx]
            f.write(f"\n\tRollout {rollout_idx} (n_defns = {rollout_res['defn_count']})")
            # convert to float for json format
            query_res = {k: round(float(v),2) for k, v in query_res.items()}
            base_res = None
            if ran_commentary: 
                trgt_resp = query_res[f'query{trgt_q_idx}']
                base_res = {k: round(float(v),2) for k, v in rollout_res['base_agg_queries'].items()}
                init_resp = base_res[f'query{trgt_q_idx}']
                
                f.write(f"\n\t\tCommentary direction: {directionality} (query {trgt_q_idx})")
                diff = trgt_resp-init_resp
                diff_str = "more" if diff > 0 else "less"
                f.write(f"\n\t\tInference moved: {diff_str}")
                f.write(f"\n\t\tDiff: {round(diff,2)}, w-comm: {trgt_resp}, init: {init_resp}")
                
                fig, ax = plt.subplots()
                # base_resps = [entry[f'query{trgt_q_idx}'] for entry in success_res[rollout_idx][f'base_query_samples']]
                # main_resps = [entry[f'query{trgt_q_idx}'] for entry in success_res[rollout_idx][f'main_query_samples']]
                base_resps = success_res[rollout_idx][f'base_query_samples'][f'query{trgt_q_idx}']
                main_resps = success_res[rollout_idx][f'main_query_samples'][f'query{trgt_q_idx}']
                agg_base = []
                agg_main=[]
                n_draw = 10
                n_individs=20
                for _ in range(n_individs): 
                    agg_base.append(np.mean(np.random.choice(base_resps, n_draw)))
                    agg_main.append(np.mean(np.random.choice(main_resps, n_draw)))
                ax.hist(agg_base, bins=10, label="Base",alpha=0.7, density=True)
                ax.hist(agg_main,bins=10, label="Commentary",alpha=0.7, density=True)
                ax.set_xlim([-0.05, 1.05]) 
                ax.legend() 
                plt.tight_layout()
                plt.savefig(f"{analysis_dir}samps_trgt_query_{rollout_idx}.png", dpi=400)
                
                
                
                
                
                
                
                #f.write(f"\n\tBase: {base_res[f'query{trgt_q_idx}']}, {base_res}")
            if len(rollout_res['unused_funcs']) != 0: 
                f.write(f"\n\t\tUnused funcs: " + json.dumps(rollout_res['unused_funcs']))
            f.write(f"\n\t\tWith commentary:{json.dumps(query_res)}\n")
            if base_res is not None: 
                f.write(f"\n\t\tInit: {json.dumps(base_res)}\n")

def load_and_process_samples_from_inference_results(inference_results_path, inference_file, who_would_win_baseline_n_samples, who_would_win_baseline_n_samples_timeout, query_7_8_take_negative, n_queries):
    with open(f"{inference_results_path}{inference_file}", "r") as f: 
        inf_data = json.load(f)
        rollout_idx = int(inference_file.split("inference_results_")[-1].split(".json")[0])
        executability = inf_data['wm']['model_metadata']['executability']
        if executability == -1: return 

        # MCMC samples 
        full_samples = inf_data['wm']['model_metadata']['official_posterior_samples']


        # who_would_win_baseline_samples = get_who_would_win_and_how_much_samples(rollout_idx=rollout_idx,
        #                         stim_dir=inference_results_path, 
        #                         override=False,
        #                         dont_recompute=False,
        #                         maximum_queries=8,
        #                         query_to_keep=8,
        #                         n_samples=who_would_win_baseline_n_samples,
        #                         timeout=who_would_win_baseline_n_samples_timeout) # load if available
        #who_would_win_transformer = get_transform_who_would_win_results(who_would_win_baseline_samples=who_would_win_baseline_samples, baseline_query='query8', take_negative=query_7_8_take_negative)

        # Full synthesis model. This could include the ablated commentary variants later on.
        samples_to_process = {"full" : full_samples}
        processed_samples = {}
        for samp_name, samples in samples_to_process.items():
            if samples is None or len(samples) == 0: continue
            if type(samples) == str: samples = eval(samples)
            
            # minimal processing of samples into query format
            # {query1: [samples], query2: [samples]}
            query_tags = set(samples[0]["value"].keys()) # all queries are computed per sample 
            if len(query_tags) != n_queries: 
                print("MISSING QUERIES: ", query_tags)
                continue
            
            query_samples = {query: [] for query in query_tags}
            for entry in samples: 
                for query, val in  entry["value"].items():
                    query_samples[query].append(val)
                    
            # process the values to make sure they are b/w 0-100
            query_samples_processed = {query: [] for query in query_samples}
            for query, values_list in query_samples.items(): 
                query_idx = int(query.split("query")[-1])
                if query_idx <= 8: 
                    if type(values_list[0]) == bool:
                        print('wrong response form: ', query_idx, values_list[0])
                        assert False
                    elif np.max(values_list) <= 1: 
                        values_list = [v*100 for v in values_list]
                
                # New match, needs to be normalized.
                # if query_idx == 7 or query_idx == 8:
                #     print('\npre transf: ', values_list[:5], values_list[-5:])
                #     values_list = transform_who_would_win_results(who_would_win_transformer=who_would_win_transformer, initial_values=values_list, take_negative=False)
                #     print('\npost transf: ',values_list[:5], values_list[-5:])
                    
                #     if np.any([x < 0 for x in values_list]): print("LESS THAN ZERO!!!")

                query_samples_processed[query] = values_list    
            processed_samples[samp_name] = query_samples_processed
    return processed_samples