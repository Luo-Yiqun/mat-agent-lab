import re
import json
import os
import numpy as np
from typing import Literal
import ast
import pandas as pd



def next_with_termination(f) -> str | None:
    try:
        return next(f)
    except StopIteration:
        return None


def parse_keywords(s: str) -> list[str]:
    return ast.literal_eval(re.search('\[[^[]+\]', s).group(0))


def parse_title(s: str, start_idx : int = 1) -> str:
    l = s.split(':')
    end = 2
    while end < len(l) and "http" not in l[end]:
        end += 1
    return ':'.join(l[start_idx: end]).strip()


def parse_link(s: str) -> str:
    assert "http" in s
    l = s.split(':')
    start = 2
    while start < len(l) and "http" not in l[start]:
        start += 1
    return ':'.join(l[start:]).strip()


def parse_value(s: str):
    if s is None or s == "None":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return s


def combine_lists(a: list, b: list) -> list:
    return a + b


def combine_dicts(a: dict, b: dict) -> dict:
    for key in b:
        if key in a:
            if type(a[key]) == type(b[key]):
                if isinstance(a[key], list):
                    a[key] = combine_lists(a[key], b[key])
                elif isinstance(a[key], dict):
                    a[key] = combine_dicts(a[key], b[key])
                else:
                    a[key] = b[key] # Use values in b, so a and b are not symmetric
            else:
                print(f"Warning: {key}'s in {a} and {b} are of different types. Will not do anything.")
        else:
            a[key] = b[key]
    return a


def parse_output(output_file: str) -> dict[dict]:
    """
    Parse the output of the keyword search + LLM literature review pipeline.
    """
    results = {}
    n_chars = []
    model = None
    rag = None
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            id = None
            results_local = {}
            for line in f:
                if "KeyWordSearch is searching results for" in line:
                    id = line.strip('.\n').split()[-1]
                    results_local = {}
                elif "Keyword" in line and "found in the" in line:
                    keywords = parse_keywords(line)
                    title = parse_title(line)
                    if title not in results_local:
                        results_local[title] = {"keywords": set()}
                    results_local[title]["keywords"].update(keywords)
                    if "http" in line:
                        if "link" not in results_local[title]:
                            results_local[title]["link"] = []
                        results_local[title]["link"].append(parse_link(line))
                elif "Warning: Scrape failed for paper" in line:
                    title = parse_title(line, start_idx = 2)
                    if title not in results_local:
                        results_local[title] = {"scrape_failed": True}
                elif "RAG is doing LLM" in line:
                # elif "RAG is doing LLM literature extraction for" in line:
                    model = line.split("with model")[1].split("and")[0].strip()
                    rag = line.split("and")[1].split("RAG")[0].strip()
                elif "of potential exp paper:" in line:
                    format = line.split("of potential exp paper:")[0].strip()
                    title = parse_title(line)
                    next_line = next_with_termination(f)
                    if next_line is None:
                        return results, n_chars
                    if "llm_exp" not in results_local[title]:
                        results_local[title]["llm_exp"] = []
                    results_local[title]["llm_exp"].append({
                        "notes": None,
                        "format": format,
                        "model": model,
                        "rag": rag
                    })
                    if "Warning: The prompt has" in next_line and "characters, which is too long. Skipping. Please manually check this paper." in next_line:
                        tmp = int(re.search('\d+', next_line).group())
                        n_chars.append(tmp)
                        results_local[title]["llm_exp"][-1]["notes"] = f"Prompt too long: {tmp} characters."
                    elif next_line.strip() == "OpenAI API error: Connection error.":
                        results_local[title]["llm_exp"][-1]["notes"] = f"OpenAI API error: Connection error."
                    elif next_line.strip() != "":
                        while next_line.strip() != '':
                            k = next_line.split(':')[0].strip()
                            v = ''.join(next_line.split(':')[1:]).strip()
                            results_local[title]["llm_exp"][-1][k] = parse_value(v)
                            next_line = next_with_termination(f)
                            if next_line is None:
                                return results, n_chars
                        results_local[title]["llm_exp"][-1][k] = parse_value(v)
                elif "of potential calc paper:" in line:
                    format = line.split("of potential calc paper:")[0].strip()
                    title = parse_title(line)
                    next_line = next_with_termination(f)
                    if next_line is None:
                        return results, n_chars
                    if "llm_calc" not in results_local[title]:
                        results_local[title]["llm_calc"] = []
                    results_local[title]["llm_calc"].append({
                        "notes": None,
                        "format": format,
                        "model": model,
                        "rag": rag
                    })
                    if "Warning: The prompt has" in next_line and "characters, which is too long. Skipping. Please manually check this paper." in next_line:
                        tmp = int(re.search('\d+', next_line).group())
                        n_chars.append(tmp)
                        results_local[title]["llm_calc"][-1]["notes"] = f"Prompt too long: {tmp} characters."
                    elif next_line.strip() == "OpenAI API error: Connection error.":
                        results_local[title]["llm_calc"][-1]["notes"] = f"OpenAI API error: Connection error."
                    elif next_line.strip() != "":
                        while next_line.strip() != '':
                            k = next_line.split(':')[0].strip()
                            v = ''.join(next_line.split(':')[1:]).strip()
                            results_local[title]["llm_calc"][-1][k] = parse_value(v)
                            next_line = next_with_termination(f)
                            if next_line is None:
                                return results, n_chars
                        results_local[title]["llm_calc"][-1][k] = parse_value(v)
                # elif "Citing papers found but no result found for" in line or "done for" in line:
                elif "All done for" in line:
                    if results_local == {}:
                        print(f"Warning: No results found for {id}")
                    for title in results_local:
                        if "keywords" in results_local[title]:
                            results_local[title]["keywords"] = list(results_local[title]["keywords"])
                    results_local["finished"] = True
                    combine_dicts(results, {id: results_local})
                    id = None
                    results_local = {}
    print(f"There are {len(results)} structures with results.")
    return results, n_chars


def get_property(json_file: str, exp_cal: Literal["llm_exp", "llm_calc"], property: str) -> dict[str, dict[str, float]]:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for id, value in data.items():
        for title, v in value.items():
            if exp_cal in v:
                if property in v[exp_cal] and v[exp_cal][property] != "None":
                    combine_dicts(result, {id: [{title: float(v[exp_cal][property])}]})
    return result


def get_property_set(json_file: str, exp_cal: Literal["llm_exp", "llm_calc"], properties: list[str]) -> tuple[set, set, int, int]:
    """
    Get the set of papers:
        LLM positive: with the properties
        LLM neutral: No LLM results
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    positive_set = set()
    neutral_set = set()
    paper_positive = 0
    paper_neutral = 0
    for id, value in data.items():
        for k, v in value.items():
            if exp_cal in v:
                if v[exp_cal] == {}:
                    neutral_set.add(id)
                    paper_neutral += 1
                else:
                    positive = True
                    for p in properties:
                        if v[exp_cal][p] == "None":
                            positive = False
                    if positive:
                        positive_set.add(id)
                        paper_positive += 1
    neutral_set = neutral_set - positive_set
    print(f"property: {exp_cal}/{properties}")
    print(f"positive set size: {len(positive_set)}")
    print(f"neutral set size: {len(neutral_set)}")
    print(f"positive papers size: {paper_positive}")
    print(f"neutral papers size: {paper_neutral}")
    print()
    return positive_set, neutral_set, paper_positive, paper_neutral


def json_to_pandas(sets: tuple[str] = ("kaiji", "sub1ev", "100-125ev", "125-150ev", "150-175ev", "175-200ev")) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keywords_exp = set(["photoemission spectroscopy", "IPES", "inverse photoemission spectroscopy",
                                "ultraviolet photoemission spectroscopy", "2PPE", "two-photon photoemission",
                                "optical gap", "band gap", "optical band gap", "singlet excitation energy",
                                "absorption", "UV-Vis",
                                "fluorescence", "photoluminescence", "fluorescent"])
    keywords_calc = set(["Green's function", "GW approximation", "Bethe-Salpeter Equation"])

    df_inaccessible = pd.DataFrame(columns = ["Set", "ID", "Title", "Notes", "Check", "Link"])
    df_exp = pd.DataFrame(columns = ["Set", "ID", "Title", "Keywords", "density_of_state", "DOS check", "absorption_spectrum", "Absorption Spectrum Check", "optical_gap", "optical_gap_source", "Optical Gap Check", "Notes", "Link"])
    df_calc = pd.DataFrame(columns = ["Set", "ID", "Title", "Keywords", "bse_gap", "bse_gap_source", "BSE Gap Check", "other_gw_bse_results", "Other GW BSE Results Check", "Notes", "link"])
    df_devices = pd.DataFrame(columns = ["Set", "ID", "Title", "SF", "TTA", "TADF", "Check", "Notes", "Link"])

    for s in sets:
        path = f"C:/Users/18000/OneDrive/Desktop/NewDataset/LiteratureReview/{s}.json"

        with open(path, "r") as f:
            data = json.load(f)

        for id, papers in data.items():
            for title, content in papers.items():
                if title == "finished":
                    continue
                if content.get("scrape_failed"):
                    df_inaccessible.loc[len(df_inaccessible)] = {"Set": s, "ID": id, "Title": title}
                    continue
                if "llm_exp" in content:
                    for extraction in content["llm_exp"]:
                        if "Prompt too long" in (extraction.get("notes", '') or ''):
                            df_inaccessible.loc[len(df_inaccessible)] = {"Set": s, "ID": id, "Title": title, "Link": link, "Notes": extraction["notes"]}
                        elif any([extraction[prop] for prop in ("density_of_state", "absorption_spectrum", "optical_gap", "optical_gap_source")]):
                            for link in content.get("link", [None]):
                                df_exp.loc[len(df_exp)] = {
                                    "Set" : s,
                                    "ID": id,
                                    "Title": title,
                                    "Keywords": keywords_exp.intersection(content.get("keywords")),
                                    "density_of_state": extraction.get("density_of_state"),
                                    "absorption_spectrum": extraction.get("absorption_spectrum"),
                                    "optical_gap": extraction.get("optical_gap"),
                                    "optical_gap_source": extraction.get("optical_gap_source"),
                                    "Link": link,
                                }
                if "llm_cal" in content:
                    for extraction in content["llm_calc"]:
                        if "Prompt too long" in (extraction.get("notes", '') or ''):
                            df_inaccessible.loc[len(df_inaccessible)] = {"Set": s, "ID": id, "Title": title, "Link": link, "Notes": extraction["notes"]}
                        elif any([extraction[prop] for prop in ("bse_gap", "bse_gap_source", "other_gw_bse_results")]):
                            for link in content.get("link", [None]):
                                df_calc.loc[len(df_calc)] = {
                                    "Set" : s,
                                    "ID": id, 
                                    "Title": title, 
                                    "Keywords": keywords_calc.intersection(content.get("keywords")),
                                    "bse_gap": extraction["bse_gap"], 
                                    "bse_gap_source": extraction["bse_gap_source"], 
                                    "other_gw_bse_results": extraction["other_gw_bse_results"],
                                    "Link": link, 
                                }
                if set(["singlet fission", "triplet-triplet annihilation", "termally activated delayed fluorescence", "TADF"]).intersection(content["keywords"]):
                    tmp = {"Set": s, "ID": id, "Title": title, "Link": None}
                    if "singlet fission" in content["keywords"]:
                        tmp["SF"] = True
                    elif "triplet-triplet annihilation" in content["keywords"]:
                        tmp["TTA"] = True
                    elif "termally activated delayed fluorescence" in content["keywords"] or "TADF" in content["keywords"]:
                        tmp["TADF"] = True
                    for link in content.get("link", [None]):
                        tmp["Link"] = link
                        df_devices.loc[len(df_devices)] = tmp

    df_inaccessible = df_inaccessible.sort_values(by=["Set", "ID"]).reset_index(drop=True)
    df_exp = df_exp.sort_values(by=["Set", "ID"]).reset_index(drop=True)
    df_calc = df_calc.sort_values(by=["Set", "ID"]).reset_index(drop=True)
    df_devices = df_devices.sort_values(by=["Set", "ID"]).reset_index(drop=True)
    
    return df_inaccessible, df_exp, df_calc, df_devices


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    results, n_chars = parse_output("LiteratureReview/sub1ev.txt")
    print(f"There are {len(results)} structures with results.")
    with open("LiteratureReview/sub1ev.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent = 4)

    """
    print(f"{sum(np.array(n_chars) < 70000)}/{len(n_chars)}", sum(np.array(n_chars) < 70000) / len(n_chars))
    print(f"{sum(np.array(n_chars) < 120000)}/{len(n_chars)}", sum(np.array(n_chars) < 120000) / len(n_chars))
    print(f"{sum(np.array(n_chars) < 300000)}/{len(n_chars)}", sum(np.array(n_chars) < 300000) / len(n_chars))
    fig, ax = plt.subplots()
    max_n_chars = max(n_chars)
    ax.hist(n_chars, bins = range(120000, max_n_chars + 1, 10000))
    ax.tick_params(labelsize = 16)
    ax.set_xlabel("Number of characters", fontsize = 18)
    ax.set_ylabel("Number of papers", fontsize = 18)
    ax.set_title("Number of characters in prompt", fontsize = 20)
    fig.tight_layout()
    plt.show()
    """
    """
    optical_gap = get_property("LiteratureReview/pah101_mini.json", "llm_exp", "optical_gap")
    optical_gap = dict(sorted(optical_gap.items(), key=lambda x: x[0]))
    for id, value in optical_gap.items():
        print(id, [list(i.values())[0] for i in value])
    """
    """
    with open("LiteratureReview/citing.json", "r", encoding="utf-8") as f:
        citations = json.load(f)
    n_citations = []
    for dir in ("pah101",):
    # for dir in ("kaiji", "sub1ev", "100-125ev", "125-150ev", "150-175ev", "175-200ev"):
        ids = [i.split(".")[0] for i in os.listdir(f"json/{dir}") if i.endswith(".json")]
        for id in ids:
            if id in citations:
                n_citations.append(len(citations[id]["citing_papers"]))
    print(dir, np.sum(n_citations), np.average(n_citations))
    """
    """
    from collections import defaultdict
    with open("LiteratureReview/pah101_paper.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    comb_to_tokens = defaultdict(list)
    for id, papers in data.items():
        for paper, values in papers.items():
            if paper != "finished":
                for result in values.get("llm_exp", []):
                    comb_to_tokens[(result["model"], result["rag"])].append(result["prompt_tokens"])
    for comb, tokens in comb_to_tokens.items():
        print(comb, np.mean(tokens))
    """
    """
    with open("LiteratureReview/citing.json", "r", encoding="utf-8") as f:
        citations = json.load(f)
    with open("LiteratureReview/pah101.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    manual_optical_gap = {
        "BENZEN": (4.69, 4.8),
        "ANTCEN": (3.16, 3.16),
        "TETCEN01": (2.38, 2.38),
        "PENCEN": (1.8, 1.85),
        "ZZZDKE01": (1.37, 1.4),
        "QQQCIG04": (2.32, 2.32),
        "QQQCIG13": (2.36, 2.36),
        "QQQCIG14": (2.31, 2.31),
        "PERLEN05": (2.58, 2.58),
        "PERLEN07": (2.49, 2.49),
        "POBPIG": (2.25, 2.25),
        "QUATER10": (1.48, 1.6),
        "CORONE01": (2.9, 2.92),
        "HBZCOR": (2.8, 2.8),
        "BEANTR": (3.14, 3.14),
        "BIPHEN": (4.1, 4.18),
        "CRYSEN01": (3.6, 3.6),
        "TERPHE02": (3.9, 3.9),
        "BNPERY": (2.4, 2.5),
        "KUBVUY": (2.9, 2.9),
        "KUBWAF01": (2.7, 2.8)
    }
    width = 0.1
    model = "gpt-5.2"
    rag = "keyword_embedding"
    n_citations, n_screened, n_extracted = 0, 0, 0
    true_positive, false_positive = 0, 0
    ids = [i.split(".")[0] for i in os.listdir(f"json/pah101") if i.endswith(".json")]
    for id in ids:
        hit = False
        optical_gap = []
        papers = set()
        if id in citations and "citing_papers" in citations[id]:
            n_citations += len(citations[id]["citing_papers"])
        if id in data:
            for paper, task in data[id].items():
                if paper != "finished" and "llm_exp" in task:
                    if task.get("llm_exp", []) != [] and not paper in papers:
                        papers.add(paper)
                        n_screened += 1
                    for search in task["llm_exp"]:
                        if search.get("model", None) == model and search.get("rag", None) == rag:
                            if any((search.get("density_of_state", None), search.get("absorption_spectrum", None), search.get("optical_gap", None))):
                                n_extracted += 1
                                if search.get("optical_gap", None):
                                    optical_gap.append(search["optical_gap"])
                                    if id in manual_optical_gap:
                                        if manual_optical_gap[id][0] - width <= search["optical_gap"] <= manual_optical_gap[id][1] + width:
                                            # print(f"Found optical gap for {id} in paper {paper}: {search['optical_gap']} eV")
                                            hit = True
                                        elif search["absorption_spectrum"] is not None:
                                            print(f"Found absorption spectrum for {id} in paper {paper}: {search['absorption_spectrum']}")
                                            pass
                                    elif id not in manual_optical_gap:
                                        hit = True
        if hit:
            if id in manual_optical_gap:
                true_positive += 1
            else:
                false_positive += 1
        elif id in manual_optical_gap:
            print(f"Missed polymorph: {id} with optical gaps: {optical_gap}")
    print(f"Number of papers per material reduced from {n_citations / 101} to {n_screened / 101} to {n_extracted / 101}")

    false_negative = len(manual_optical_gap) - true_positive
    true_negative = 101 - len(manual_optical_gap) - false_positive
    print(f"True positives: {true_positive}")
    print(f"False negatives: {false_negative}")
    print(f"False positives: {false_positive}")
    print(f"True negatives: {true_negative}")
    accuracy = (true_positive + true_negative) / 101
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    """
    df_inaccessible, df_exp, df_calc, df_devices = json_to_pandas()
    df_inaccessible.to_csv("LiteratureReview/inaccessible.csv", index=False)
    df_exp.to_csv("LiteratureReview/exp.csv", index=False)
    df_calc.to_csv("LiteratureReview/calc.csv", index=False)
    df_devices.to_csv("LiteratureReview/devices.csv", index=False)
