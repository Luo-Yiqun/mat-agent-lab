"""
@ Yiqun Luo (luo2@andrew.cmu.edu)

This script is used to search for certain properties of the citing papers of a given CSD reference code.

Please run the code under the campus network for access to databases.
The current keyword search version needs manually extracting data, an LLM would further automate that.

STRONGLY recommend to use selenium to avoid access forbidden (403) error, although it cannot be avoided sometimes.
If you want to use selenium, please download and install the chromedriver and set the path to the chromedriver.
You may need to manually answer robot verification, especially for logining to Google Scholar for the first time.

Please put the config.json containing the OpenAI API under the LiteratureReview dataset!

Todo:
get PDF link, maybe Google search title
"""

# Standard library imports
import json
from typing import Literal
import time
import sys
import re

# Third-party imports
from tqdm import tqdm

import trafilatura
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

# Local imports
from citer import CCDCCitingPaper
from reviewer import KeyWordSearch, LLMReviewer
from utilities import *



# Helpers to build prompts and optionally run RAG consistently for PDF/HTML
def base_prompt(
        id: str,
        c: CCDCCitingPaper,
        exp_calc: Literal["exp", "calc"]
        ) -> str:
    return (
        f"{getattr(LLMReviewer, f'second_prompt_{exp_calc}')}\n\n"
        f"CSD reference code of the crystal of interest: {id}\n"
        f"SMILES of the crystal of interest: {c.entry.molecule.smiles}\n\n"
    )


def prepare_prompt_with_content(
    id: str,
    paper: dict,
    c: CCDCCitingPaper,

    exp_calc: Literal["exp", "calc"],
    content: str,
    llm_reviewer: LLMReviewer,
    rag: Literal[None, "keyword", "keyword_embedding"],
    fmt: Literal["PDF", "HTML"],
    keywords: list[str],

    output_txt: str = "LiteratureReview/output.txt",
) -> str:
    prompt = base_prompt(id, c, exp_calc)
    rag_snippets = []
    if rag == "keyword":
        if fmt == "HTML":
            content = trafilatura.extract(content, favor_recall = True, include_comments = True, include_images = True, include_tables = True)
        rag_snippets = llm_reviewer.rag_keyword_only(content, keywords)
        if rag_snippets == []:
            with open(output_txt, "a", encoding="utf-8") as f:
                f.write(f"Keyword-only RAG did not find any relevant content for paper {paper['title']}.\n\n")
            return ""
        else:
            prompt += f"Paper {fmt} content after keyword-based retrival augmented generation:\n"
            for i, snippet in enumerate(rag_snippets, start = 1):
                prompt += f"Snippet {i}:\n{snippet}\n\n"
    elif rag == "keyword_embedding":
        rag_snippets = llm_reviewer.rag_keyword_and_embedding(
            text=content,
            format=fmt,
            keywords=keywords,
            query=getattr(LLMReviewer, f"static_prompt_{exp_calc}") + getattr(LLMReviewer, f"second_prompt_{exp_calc}"),
        )
        if rag_snippets == []:
            with open(output_txt, "a", encoding="utf-8") as f:
                f.write(f"Keyword-embedding RAG did not find any relevant content for paper {paper['title']}.\n\n")
            return ""
        else:
            prompt += f"Paper {fmt} content after keyword- and embedding-space-based retrival augmented generation:\n"
            for i, snippet in enumerate(rag_snippets, start = 1):
                prompt += f"Snippet {i}:\n{snippet}\n\n"
    if rag is None or rag_snippets == []:
        if fmt == "HTML":
            content = trafilatura.extract(content, favor_recall = True, include_comments = True, include_images = True, include_tables = True)
        prompt += f"Paper {fmt} content:\n{content}"
    return prompt


def keyword_llm_paper(
    id: str,
    paper: dict,
    driver: webdriver.Chrome,
    rag: Literal[None, "keyword", "keyword_embedding"] = "keyword",
    citing_json: str = "LiteratureReview/citing.json",
    output_txt: str = "LiteratureReview/output.txt",
    llm_model: str = "gpt-5-mini",
):
    with open(output_txt, "a", encoding="utf-8") as f:
        f.write('\n')

    # Get the citing papers, including the original paper
    c = CCDCCitingPaper(id, driver = driver, json_file = citing_json)
    # c.get_original_papers()
    # c.citer()

    # First round: screen the citing papers using keyword
    print(f"Starting keyword review for paper: {paper['google_scholar_title']}...")
    keyword_reviewer = KeyWordSearch(output_file = output_txt, driver = driver)
    keyword_reviewer.search_one(paper)
    print("Keyword search score:", paper.get("score", 0))

    # Second round: extract the experimental results using LLM
    print(f"Starting LLM review for paper: {paper['google_scholar_title']}...")
    llm_reviewer = LLMReviewer(id = id, output_file = output_txt, model2 = llm_model, driver = driver)
    with open(output_txt, "a", encoding="utf-8") as f:
        f.write(f"{LLMReviewer.__name__} with model {llm_reviewer.model2} and {rag} RAG is doing LLM literature extraction for paper: {paper['google_scholar_title']}...\n")
        f.write("--------------------------------\n")
    
    for exp_calc in ["exp", "calc"]:
        if not set(paper["keywords"]) & set(getattr(keyword_reviewer, f"keywords_{exp_calc}")):
            continue
        reply = ''
        if reply == '' and "pdf_link_content" in paper and paper["pdf_link_content"]:
            with open(output_txt, "a", encoding="utf-8") as f:
                f.write(f"PDF of potential {exp_calc} paper: {paper['google_scholar_title']}: {paper['pdf_link']}\n")
            try:
                prompt = prepare_prompt_with_content(
                    id = id,
                    paper = paper,
                    c = c,
                    exp_calc = exp_calc,
                    content = paper['pdf_link_content'],
                    llm_reviewer = llm_reviewer,
                    rag = rag,
                    fmt = "PDF",
                    keywords = paper["keywords"],
                    output_txt = output_txt,
                )
                reply, tokens = llm_reviewer.search_text(llm_reviewer.model2, exp_calc, prompt, getattr(LLMReviewer, f"second_prompt_{exp_calc}_schema"))
                reply_json = json.loads(reply)
            except ValueError as e:
                print(f"RAG failed for paper {paper['google_scholar_title']}: {e}")
            except json.JSONDecodeError as e:
                print(f"Could not parse JSON from model 2 reply: {reply}\nError: {e}")
            else:
                with open(output_txt, "a", encoding="utf-8") as f:
                    for k, v in tokens.items():
                        f.write(f"{k}: {v}\n")
                    for k, v in reply_json.items():
                        k_clean = re.sub("\n", "", str(k))
                        v_clean = re.sub("\n", "", str(v))
                        f.write(f"{k_clean}: {v_clean}\n")
                    f.write('\n')
        if reply == '' and "publication_link_content" in paper and paper["publication_link_content"]:
            with open(output_txt, "a", encoding="utf-8") as f:
                f.write(f"PUBLICATION WEBSITE of potential {exp_calc} paper: {paper['google_scholar_title']}: {paper['publication_link']}\n")
            try:
                prompt = prepare_prompt_with_content(
                    id = id,
                    paper = paper,
                    c = c,
                    exp_calc = exp_calc,
                    content = paper["publication_link_content"],
                    llm_reviewer = llm_reviewer,
                    rag = rag,
                    fmt = "HTML",
                    keywords = paper["keywords"],
                    output_txt = output_txt,
                )
                reply, tokens = llm_reviewer.search_text(llm_reviewer.model2, exp_calc, prompt, getattr(LLMReviewer, f"second_prompt_{exp_calc}_schema"))
                reply_json = json.loads(reply)
            except ValueError as e:
                print(f"RAG failed for paper {paper['google_scholar_title']}: {e}")
            except json.JSONDecodeError as e:
                print(f"Could not parse JSON from model 2 reply: {reply}\nError: {e}")
            else:
                with open(output_txt, "a", encoding="utf-8") as f:
                    for k, v in tokens.items():
                        f.write(f"{k}: {v}\n")
                    for k, v in reply_json.items():
                        k_clean = re.sub("\n", "", str(k))
                        v_clean = re.sub("\n", "", str(v))
                        f.write(f"{k_clean}: {v_clean}\n")
                    f.write('\n')
    
    with open(output_txt, "a", encoding="utf-8") as f:
        f.write("--------------------------------\n")
        f.write(f"{LLMReviewer.__name__} with model {llm_reviewer.model2} and {rag} RAG is done LLM review for paper: {paper['google_scholar_title']}.\n")
    
    return


def keyword_llm_pipeline(
        ids: list[str],
        output_txt: str = "LiteratureReview/output.txt",
        citing_json: str = "LiteratureReview/citing.json",
        llm_model: str = "gpt-5-mini",
        rag: Literal[None, "keyword", "keyword_embedding"] = "keyword"
        ):
    """
    A pipeline to review the citing papers using keyword and LLM.

    Total cost: $46.74
    """

    visited, _ = parse_output(output_txt)
    visited = [k for k, v in visited.items() if v.get("finished", False)]
    start_idx = 0
    step = 100
    print(f"Already visited {len(visited)}/{len(ids)} polymorphs: {visited}")

    start_time = time.time()
    service = Service("C:/Users/18000/OneDrive/Desktop/VSCode/chromedriver-win64/chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    for id in ids:
        if id in visited:
            continue
        print(f"Starting pipeline for {id}...")
        with open(output_txt, "a", encoding="utf-8") as f:
            f.write('\n')

        # Get citing papers, including the original paper
        c = CCDCCitingPaper(id, driver = driver, json_file = citing_json)
        c.get_original_papers()
        c.citer(False)
        print(f"{len(c.citing_papers)} citing papers found for {id}.")
        print()

        while start_idx < len(c.citing_papers):
            print(f"Processing papers {start_idx} to {min(start_idx + step, len(c.citing_papers))} for {id}...")

            # First round: screen the citing papers using keyword
            print(f"Starting keyword review for {id}...")
            keyword_reviewer = KeyWordSearch(id, output_txt, driver = driver)
            keyword_reviewer.search(c.citing_papers[start_idx: start_idx + step])
            print()

            # Second round: extract experimental results using LLM
            print(f"Starting LLM review for {id}...")
            llm_reviewer = LLMReviewer(id, output_txt, model2 = llm_model)
            with open(output_txt, "a", encoding="utf-8") as f:
                f.write(f"{LLMReviewer.__name__} with model {llm_reviewer.model2} and {rag} RAG is doing LLM literature extraction for {id}...\n")
                f.write("--------------------------------\n")

            for paper in tqdm(c.citing_papers[start_idx: start_idx + step], smoothing = 0.1, ncols = 80):
                if paper["score"] > 0.7:
                    for exp_calc in ["exp", "calc"]:
                        if not set(paper["keywords"]) & set(getattr(keyword_reviewer, f"keywords_{exp_calc}")):
                            continue
                        reply = ""
                        if reply == "" and "pdf_link_content" in paper and paper["pdf_link_content"] != '':
                            with open(output_txt, "a", encoding="utf-8") as f:
                                f.write(f"PDF of potential {exp_calc} paper: {paper['google_scholar_title']}: {paper['pdf_link']}\n")
                            try:
                                prompt = prepare_prompt_with_content(
                                    id = id,
                                    paper = paper,
                                    c = c,
                                    exp_calc = exp_calc,
                                    content = paper['pdf_link_content'],
                                    llm_reviewer = llm_reviewer,
                                    rag = rag,
                                    fmt = "PDF",
                                    keywords = paper["keywords"],
                                    output_txt = output_txt,
                                )
                                if prompt == "":
                                    raise ValueError("No relevant content found in RAG.")
                                reply, tokens = llm_reviewer.search_text(llm_reviewer.model2, exp_calc, prompt, getattr(llm_reviewer, f"second_prompt_{exp_calc}_schema"))
                                reply_json = json.loads(reply)
                            except ValueError as e:
                                print(f"RAG failed for paper {paper['google_scholar_title']}: {e}")
                            except json.JSONDecodeError as e:
                                print(f"Could not parse JSON from model 2 reply: {reply}\nError: {e}")
                            else:
                                with open(output_txt, "a", encoding="utf-8") as f:
                                    for k, v in tokens.items():
                                        f.write(f"{k}: {v}\n")
                                    for k, v in reply_json.items():
                                        k_clean = re.sub("\n", "", str(k))
                                        v_clean = re.sub("\n", "", str(v))
                                        f.write(f"{k_clean}: {v_clean}\n")
                                    f.write('\n')
                        if reply == "" and "publication_link_content" in paper and paper["publication_link_content"] != '':
                            with open(output_txt, "a", encoding="utf-8") as f:
                                f.write(f"PUBLICATION WEBSITE of potential {exp_calc} paper: {paper['google_scholar_title']}: {paper['publication_link']}\n")
                            try:
                                prompt = prepare_prompt_with_content(
                                    id = id,
                                    paper = paper,
                                    c = c,
                                    exp_calc = exp_calc,
                                    content = paper["publication_link_content"],
                                    llm_reviewer = llm_reviewer,
                                    rag = rag,
                                    fmt = "HTML",
                                    keywords = paper["keywords"],
                                    output_txt = output_txt,
                                )
                                if prompt == "":
                                    raise ValueError("No relevant content found in RAG.")
                                reply, tokens = llm_reviewer.search_text(llm_reviewer.model2, exp_calc, prompt, getattr(llm_reviewer, f"second_prompt_{exp_calc}_schema"))
                                reply_json = json.loads(reply)
                            except ValueError as e:
                                print(f"RAG failed for paper {paper['google_scholar_title']}: {e}")
                            except json.JSONDecodeError as e:
                                print(f"Could not parse JSON from model 2 reply: {reply}\nError: {e}")
                            else:
                                with open(output_txt, "a", encoding="utf-8") as f:
                                    for k, v in tokens.items():
                                        f.write(f"{k}: {v}\n")
                                    for k, v in reply_json.items():
                                        k_clean = re.sub("\n", "", str(k))
                                        v_clean = re.sub("\n", "", str(v))
                                        f.write(f"{k_clean}: {v_clean}\n")
                                    f.write('\n')
            with open(output_txt, "a", encoding="utf-8") as f:
                f.write("--------------------------------\n")
                f.write(f"{LLMReviewer.__name__} with model {llm_reviewer.model2} and {rag} RAG is done LLM for {id}.\n")
                if rag == "keyword_embedding":
                    f.write(f"RAG stats: {llm_reviewer.rag_stats}\n")
            
            start_idx += step
            if start_idx >= len(c.citing_papers):
                message = f"All done for {id}.\n"
                with open(output_txt, "a", encoding="utf-8") as f:
                    f.write(message)
                print(message)
            else:
                message = f"{start_idx}/{len(c.citing_papers)} papers done for {id}.\n"
                with open(output_txt, "a", encoding="utf-8") as f:
                    f.write(message)
                print(message)

        start_idx = 0
    
    driver.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total elapsed time: {hours:02d}h {minutes:02d}m {seconds:02d}s")

    return


def pah101_paper(
    output_txt: str = "LiteratureReview/pah101_paper.txt"
):
    citing_json: str = "LiteratureReview/citing.json"
    llm_models = ("gpt-5-mini", "gpt-5", "gpt-5.2")
    rags = (None, "keyword", "keyword_embedding")

    id_to_papers = {
        "ZZZDKE01": [
            {
                "publication_link": "https://www.nature.com/articles/nchem.1381",
                "google_scholar_title": "The synthesis, crystal structure and charge-transport properties of hexacene",
                "google_scholar_snippet": "Acenes can be thought of as one-dimensional strips of graphene and they have the potential to be used in the next generation of electronic devices. However, because acenes larger than pentacene have been found to be unstable, it was generally accepted that they would not be particularly useful materials under normal conditions. Here, we show that, by using a physical vapour-transport method, platelet-shaped crystals of hexacene can be prepared from a monoketone precursor. These crystals are stable in the dark for a long period of time …",
                "pdf_link": "https://www.nature.com/articles/nchem.1381.pdf"
            },
            {
                "publication_link": "https://pubs.acs.org/doi/abs/10.1021/ja503980c",
                "google_scholar_title": "Multiphonon relaxation slows singlet fission in crystalline hexacene",
                "google_scholar_snippet": "Singlet fission, the conversion of a singlet excitation into two triplet excitations, is a viable route to improved solar-cell efficiency. Despite active efforts to understand the singlet fission …",
                "pdf_link": "https://pubs.acs.org/doi/pdf/10.1021/ja503980c"
            },
        ],
        "QQQCIG13": [
            {
                "publication_link": "https://pubs.rsc.org/en/content/articlehtml/2010/jm/b914334c",
                "pdf_link": "https://pubs.rsc.org/en/content/articlepdf/2010/jm/b914334c",
                "google_scholar_title": "Rubrene micro-crystals from solution routes: their crystallography, morphology and optical properties",
                "google_scholar_snippet": "A series of rubrene micro-crystals (MCs) with controllable sizes and shapes, ranging from one-dimensional (1D) ribbons to 2D rhombic and hexagonal plates, have bee prepared by employing the reprecipitation method. Based on X-ray diffraction analysis, the crystal structures of 1D ribbons and 2D plates have been identified to be triclinic and monoclinic phases, respectively, rather than the commonly reported orthorhombic phase for vacuum-deposited rubrene crystals. In our system, adjustment of the monomer concentration of …"
            }
        ],
        "QQQCIG14": [
            {
                "publication_link": "https://pubs.rsc.org/en/content/articlehtml/2010/jm/b914334c",
                "pdf_link": "https://pubs.rsc.org/en/content/articlepdf/2010/jm/b914334c",
                "google_scholar_title": "Rubrene micro-crystals from solution routes: their crystallography, morphology and optical properties",
                "google_scholar_snippet": "A series of rubrene micro-crystals (MCs) with controllable sizes and shapes, ranging from one-dimensional (1D) ribbons to 2D rhombic and hexagonal plates, have bee prepared by employing the reprecipitation method. Based on X-ray diffraction analysis, the crystal structures of 1D ribbons and 2D plates have been identified to be triclinic and monoclinic phases, respectively, rather than the commonly reported orthorhombic phase for vacuum-deposited rubrene crystals. In our system, adjustment of the monomer concentration of …"
            }
        ],
        "HBZCOR": [
            {
                "publication_link": "https://journals.aps.org/prb/abstract/10.1103/PhysRevB.63.205409",
                "google_scholar_title": "Comparison of ultraviolet photoelectron spectroscopy and scanning tunneling spectroscopy measurements on highly ordered ultrathin films of hexa-peri …",
                "google_scholar_snippet": "Abstract Hexa-peri-hexabenzocoronene (C 42 H 18, HBC) films adsorbed on the Au (111) surface were investigated by means of ultraviolet photoelectron spectroscopy (UPS) and …",
                "pdf_link": "https://scholar.archive.org/work/phdzodk52za6fntkz45ypxhgoq/access/wayback/http://www.iapp.de:80/~paper/paper/ombe/Comparison%20of%20UPS%20and%20STS%20measurements%20on%20highly%20ordered%20ultrathin%20films.pdf"
            }
        ],
        "BIPHEN": [
            {
                "publication_link": "https://www.tandfonline.com/doi/abs/10.1080/00268979100100411",
                "pdf_link": "https://www.tandfonline.com/doi/pdf/10.1080/00268979100100411",
                "google_scholar_title": "Optical and magnetic properties of the exact PPP states of biphenyl",
                "google_scholar_snippet": "The low-lying singlets and triplets of biphenyl are obtained exactly within the PPP model using the diagrammatic valence bond method. The energy gaps within the singlet manifold …"
            },
            {
                "publication_link": "https://www.sciencedirect.com/science/article/pii/S0379677900004318",
                "pdf_link": "https://www.sciencedirect.com/science/article/pii/S0379677900004318",
                "google_scholar_title": "[HTML][HTML] Pressure studies on the intermolecular interactions in biphenyl",
                "google_scholar_snippet": "We investigate the influence of intermolecular interactions on the optical and structural properties of oligophenyls in solid films of polycrystalline nature. To this end, we have …"
            },
            {
                "publication_link": "https://pubs.aip.org/aip/jcp/article-abstract/41/12/3928/80881",
                "pdf_link": "https://pubs.aip.org/aip/jcp/article-pdf/41/12/3928/18836564/3928_1_online.pdf",
                "google_scholar_title": "Electronic structure and spectra of biphenyl and its related compound",
                "google_scholar_snippet": "The Pariser—Parr method has been used to elucidate the electronic structure and the spectrum of biphenyl. The absorptions at 300, 250, 200, and 170 mμ seem to be explained …"
            }
        ],
        "TERPHE02": [
            {
                "publication_link": "https://www.sciencedirect.com/science/article/pii/0301010482870134",
                "google_scholar_title": "Absorption spectra of volatile aromatic hydrocarbon films in the vacuum ultraviolet region",
                "google_scholar_snippet": "Absorption spectra of volatile aromatic hydrocarbon films, p-terphenyl, chrysene, benzanthracene and triphenylene, in the near and vacuum ultraviolet region down to 130 …",
                "pdf_link": ""
            }
        ],
        "KUBVUY": [
            {
                "publication_link": "https://pubs.rsc.org/en/content/articlehtml/2017/sc/c8tc05817b",
                "pdf_link": "https://pubs.rsc.org/en/content/getauthorversionpdf/c8tc05817b",
                "google_scholar_title": "Absence of delayed fluorescence and triplet–triplet annihilation in organic light emitting diodes with spatially orthogonal bianthracenes",
                "google_scholar_snippet": "Two compounds, 2-methyl-9, 10-bis (naphthalen-2-yl) anthracene (MADN), which has a single anthracene unit, and 10, 10′-diphenyl-9, 9'-bianthracene (PPBA), which has two spatially orthogonal anthracene units, were compared and investigated in terms of photoelectric characteristics and the reverse intersystem crossing (RISC) process in organic light emitting diodes (OLEDs). Transient electroluminescence (EL) measurements indicated large contributions of triplet–triplet annihilation (TTA) for MADN but almost no contribution of …"
            }
        ],
        "KUBWAF": [
            {
                "publication_link": "https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.2c03356",
                "google_scholar_title": "Comparative Study of Exciton Dynamics in 9, 9′-Bianthracene Nanoaggregates and Thin Films: Observation of Singlet–Singlet Annihilation-Mediated Triplet Exciton …",
                "google_scholar_snippet": "Steady-state and time-resolved photophysical studies are performed with freshly prepared nanoaggregates (Bi-An NA) and vapor-deposited thin films (Bi-An TF) of 9, 9′-bianthracene …",
                "pdf_link": ""
            }
        ]
    }

    start_time = time.time()
    service = Service("C:/Users/18000/OneDrive/Desktop/VSCode/chromedriver-win64/chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    for id, papers in id_to_papers.items():
        if id != "TERPHE02":
            continue
        print(f"Starting pipeline for {id}...")
        with open(output_txt, "a", encoding="utf-8") as f:
            f.write('\n')
        
        # Get the citing papers, including the original paper
        c = CCDCCitingPaper(id, driver = driver, json_file = citing_json)

        # First round: screen the citing papers using keyword
        print(f"Starting keyword review for {id}...")
        keyword_reviewer = KeyWordSearch(id, output_txt, driver = driver)
        keyword_reviewer.search(papers)
        print()

        # Second round: extract the experimental results using LLM
        print(f"Starting LLM review for {id}...")
        for llm_model in llm_models:
            print(f"Using LLM model: {llm_model} with keyword RAG...")
            llm_reviewer = LLMReviewer(id, output_txt, model2 = llm_model)
            with open(output_txt, "a", encoding="utf-8") as f:
                f.write(f"{LLMReviewer.__name__} with model {llm_reviewer.model2} and keyword RAG is doing LLM literature extraction for {id}...\n")
                f.write("--------------------------------\n")
            
            for paper in tqdm(papers, smoothing = 0.1, ncols = 80):
                for exp_calc in ("exp", "calc"):
                    if not set(paper["keywords"]) & set(getattr(keyword_reviewer, f"keywords_{exp_calc}")):
                        continue
                    reply = ''
                    if reply == '' and "pdf_link_content" in paper and paper["pdf_link_content"] and paper["pdf_link_content"]:
                        with open(output_txt, "a", encoding="utf-8") as f:
                            f.write(f"PDF of potential {exp_calc} paper: {paper['google_scholar_title']}: {paper['pdf_link']}\n")
                        try:
                            prompt = prepare_prompt_with_content(
                                id = id,
                                paper = paper,
                                c = c,
                                exp_calc = exp_calc,
                                content = paper['pdf_link_content'],
                                llm_reviewer = llm_reviewer,
                                rag = "keyword",
                                fmt = "PDF",
                                keywords = paper["keywords"],
                                output_txt = output_txt,
                            )
                            if prompt == "":
                                raise ValueError("No relevant content found in RAG.")
                            reply, tokens = llm_reviewer.search_text(llm_reviewer.model2, exp_calc, prompt, getattr(llm_reviewer, f"second_prompt_{exp_calc}_schema"))
                            reply_json = json.loads(reply)
                        except ValueError as e:
                            print(f"RAG failed for paper {paper['google_scholar_title']}: {e}")
                        except json.JSONDecodeError as e:
                            print(f"Could not parse JSON from model 2 reply: {reply}\nError: {e}")
                        else:
                            with open(output_txt, "a", encoding="utf-8") as f:
                                for k, v in tokens.items():
                                    f.write(f"{k}: {v}\n")
                                for k, v in reply_json.items():
                                    k_clean = re.sub("\n", "", str(k))
                                    v_clean = re.sub("\n", "", str(v))
                                    f.write(f"{k_clean}: {v_clean}\n")
                                f.write('\n')
                    if reply == '' and "publication_link_content" in paper and paper["publication_link_content"] and paper["publication_link_content"]:
                        with open(output_txt, "a", encoding="utf-8") as f:
                            f.write(f"PUBLICATION WEBSITE of potential {exp_calc} paper: {paper['google_scholar_title']}: {paper['publication_link']}\n")
                        try:
                            prompt = prepare_prompt_with_content(
                                id = id,
                                paper = paper,
                                c = c,
                                exp_calc = exp_calc,
                                content = paper["publication_link_content"],
                                llm_reviewer = llm_reviewer,
                                rag = "keyword",
                                fmt = "HTML",
                                keywords = paper["keywords"],
                                output_txt = output_txt,
                            )
                            if prompt == "":
                                raise ValueError("No relevant content found in RAG.")
                            reply, tokens = llm_reviewer.search_text(llm_reviewer.model2, exp_calc, prompt, getattr(llm_reviewer, f"second_prompt_{exp_calc}_schema"))
                            reply_json = json.loads(reply)
                        except ValueError as e:
                            print(f"RAG failed for paper {paper['google_scholar_title']}: {e}")
                        except json.JSONDecodeError as e:
                            print(f"Could not parse JSON from model 2 reply: {reply}\nError: {e}")
                        else:
                            with open(output_txt, "a", encoding="utf-8") as f:
                                for k, v in tokens.items():
                                    f.write(f"{k}: {v}\n")
                                for k, v in reply_json.items():
                                    k_clean = re.sub("\n", "", str(k))
                                    v_clean = re.sub("\n", "", str(v))
                                    f.write(f"{k_clean}: {v_clean}\n")
                                f.write('\n')

            with open(output_txt, "a", encoding="utf-8") as f:
                f.write("--------------------------------\n")
                f.write(f"{LLMReviewer.__name__} with model {llm_reviewer.model2} and keyword RAG is done LLM for {id}.\n")
        
        for rag in rags:
            if rag == "keyword":
                continue
            print(f"Using LLM model: gpt-5.2 with {rag} RAG...")
            llm_reviewer = LLMReviewer(id, output_txt, model2 = "gpt-5.2")
            with open(output_txt, "a", encoding="utf-8") as f:
                f.write(f"{LLMReviewer.__name__} with model {llm_reviewer.model2} and {rag} RAG is doing LLM literature extraction for {id}...\n")
                f.write("--------------------------------\n")
            
            for paper in tqdm(papers, smoothing = 0.1, ncols = 80):
                for exp_calc in ("exp", "calc"):
                    if not set(paper["keywords"]) & set(getattr(keyword_reviewer, f"keywords_{exp_calc}")):
                        continue
                    reply = ''
                    if reply == '' and "pdf_link_content" in paper and paper["pdf_link_content"] and paper["pdf_link_content"]:
                        with open(output_txt, "a", encoding="utf-8") as f:
                            f.write(f"PDF of potential {exp_calc} paper: {paper['google_scholar_title']}: {paper['pdf_link']}\n")
                        try:
                            prompt = prepare_prompt_with_content(
                                id = id,
                                paper = paper,
                                c = c,
                                exp_calc = exp_calc,
                                content = paper['pdf_link_content'],
                                llm_reviewer = llm_reviewer,
                                rag = rag,
                                fmt = "PDF",
                                keywords = paper["keywords"],
                                output_txt = output_txt,
                            )
                            if prompt == "":
                                raise ValueError("No relevant content found in RAG.")
                            reply, tokens = llm_reviewer.search_text(llm_reviewer.model2, exp_calc, prompt, getattr(llm_reviewer, f"second_prompt_{exp_calc}_schema"))
                            reply_json = json.loads(reply)
                        except ValueError as e:
                            print(f"RAG failed for paper {paper['google_scholar_title']}: {e}")
                        except json.JSONDecodeError as e:
                            print(f"Could not parse JSON from model 2 reply: {reply}\nError: {e}")
                        else:
                            with open(output_txt, "a", encoding="utf-8") as f:
                                for k, v in tokens.items():
                                    f.write(f"{k}: {v}\n")
                                for k, v in reply_json.items():
                                    k_clean = re.sub("\n", "", str(k))
                                    v_clean = re.sub("\n", "", str(v))
                                    f.write(f"{k_clean}: {v_clean}\n")
                                f.write('\n')
                    if reply == '' and "publication_link_content" in paper and paper["publication_link_content"] and paper["publication_link_content"]:
                        with open(output_txt, "a", encoding="utf-8") as f:
                            f.write(f"PUBLICATION WEBSITE of potential {exp_calc} paper: {paper['google_scholar_title']}: {paper['publication_link']}\n")
                        try:
                            prompt = prepare_prompt_with_content(
                                id = id,
                                paper = paper,
                                c = c,
                                exp_calc = exp_calc,
                                content = paper["publication_link_content"],
                                llm_reviewer = llm_reviewer,
                                rag = rag,
                                fmt = "HTML",
                                keywords = paper["keywords"],
                                output_txt = output_txt,
                            )
                            if prompt == "":
                                raise ValueError("No relevant content found in RAG.")
                            reply, tokens = llm_reviewer.search_text(llm_reviewer.model2, exp_calc, prompt, getattr(llm_reviewer, f"second_prompt_{exp_calc}_schema"))
                            reply_json = json.loads(reply)
                        except ValueError as e:
                            print(f"RAG failed for paper {paper['google_scholar_title']}: {e}")
                        except json.JSONDecodeError as e:
                            print(f"Could not parse JSON from model 2 reply: {reply}\nError: {e}")
                        else:
                            with open(output_txt, "a", encoding="utf-8") as f:
                                for k, v in tokens.items():
                                    f.write(f"{k}: {v}\n")
                                for k, v in reply_json.items():
                                    k_clean = re.sub("\n", "", str(k))
                                    v_clean = re.sub("\n", "", str(v))
                                    f.write(f"{k_clean}: {v_clean}\n")
                                f.write('\n')

            with open(output_txt, "a", encoding="utf-8") as f:
                f.write("--------------------------------\n")
                f.write(f"{LLMReviewer.__name__} with model gpt-5.2 and {rag} RAG is done LLM for {id}.\n")
                if rag == "keyword_embedding":
                    f.write(f"RAG stats: {llm_reviewer.rag_stats}\n")

        message = f"All done for {id}.\n"
        with open(output_txt, "a", encoding="utf-8") as f:
            f.write(message)
        print(message)
    
    driver.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total elapsed time: {hours:02d}h {minutes:02d}m {seconds:02d}s")

    return



def pah101_pipeline(
    output_txt : str = "LiteratureReview/pah101.txt",
    start: int = 0,
    end: int = 101
):
    """
    Start cost: $0.18
    end cost: $512.85
    """
    citing_json: str = "LiteratureReview/citing.json"
    llm_models = ("gpt-5-mini", "gpt-5", "gpt-5.2")
    rags = (None, "keyword", "keyword_embedding")
    # llm_models = ("gpt-5-mini",)
    # rags = ("keyword",)

    ids = [i.split('.')[0] for i in os.listdir("json/pah101/")][start: end]
    visited, _ = parse_output(output_txt)
    visited = [k for k, v in visited.items() if v.get("finished", False)]
    start_idx = 0
    step = 100
    print(f"Already visited {len(visited)} polymorphs: {visited}")

    start_time = time.time()
    service = Service("C:/Users/18000/OneDrive/Desktop/VSCode/chromedriver-win64/chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    for id in ids:
    # for id in ["KUBWAF01", "BEANTR", "BNPERY"]:
        if id in visited:
            continue
        print(f"Starting pipeline for {id}...")
        with open(output_txt, "a", encoding="utf-8") as f:
            f.write('\n')
        
        # Get citing papers, including the original paper
        c = CCDCCitingPaper(id, driver = driver, json_file = citing_json)
        c.get_original_papers()
        c.citer(False)
        print(f"{len(c.citing_papers)} citing papers found for {id}.")
        print()

        while start_idx < len(c.citing_papers):
            print(f"Processing papers {start_idx} to {min(start_idx + step, len(c.citing_papers))} for {id}...")

            # First round: screen the citing papers using keyword
            print(f"Starting keyword review for {id}...")
            keyword_reviewer = KeyWordSearch(id, output_txt, driver = driver)
            keyword_reviewer.search(c.citing_papers[start_idx : start_idx + step])
            print()

            # Second round: extract experimental results using LLM
            print(f"Starting LLM review for {id}...")
            for llm_model in llm_models:
                print(f"Using LLM model: {llm_model} with keyword RAG...")
                llm_reviewer = LLMReviewer(id, output_txt, model2 = llm_model)
                with open(output_txt, "a", encoding="utf-8") as f:
                    f.write(f"{LLMReviewer.__name__} with model {llm_reviewer.model2} and keyword RAG is doing LLM literature extraction for {id}...\n")
                    f.write("--------------------------------\n")

                for paper in tqdm(c.citing_papers[start_idx : start_idx + step], smoothing = 0.1, ncols = 80):
                    if paper["score"] > 0.7:
                        for exp_calc in ["exp", "calc"]:
                            if not set(paper["keywords"]) & set(getattr(keyword_reviewer, f"keywords_{exp_calc}")):
                                continue
                            reply = ''
                            if reply == '' and "pdf_link_content" in paper and paper["pdf_link_content"] and paper["pdf_link_content"] != '':
                                with open(output_txt, "a", encoding="utf-8") as f:
                                    f.write(f"PDF of potential {exp_calc} paper: {paper['google_scholar_title']}: {paper['pdf_link']}\n")
                                try:
                                    prompt = prepare_prompt_with_content(
                                        id = id,
                                        paper = paper,
                                        c = c,
                                        exp_calc = exp_calc,
                                        content = paper['pdf_link_content'],
                                        llm_reviewer = llm_reviewer,
                                        rag = "keyword",
                                        fmt = "PDF",
                                        keywords = paper["keywords"],
                                        output_txt = output_txt,
                                    )
                                    if prompt == "":
                                        raise ValueError("No relevant content found in RAG.")
                                    reply = llm_reviewer.search_text(llm_reviewer.model2, exp_calc, prompt, getattr(llm_reviewer, f"second_prompt_{exp_calc}_schema"))
                                    reply_json = json.loads(reply)
                                except ValueError as e:
                                    print(f"RAG failed for paper {paper['google_scholar_title']}: {e}")
                                except json.JSONDecodeError as e:
                                    print(f"Could not parse JSON from model 2 reply: {reply}\nError: {e}")
                                else:
                                    with open(output_txt, "a", encoding="utf-8") as f:
                                        for k, v in reply_json.items():
                                            k_clean = re.sub("\n", "", str(k))
                                            v_clean = re.sub("\n", "", str(v))
                                            f.write(f"{k_clean}: {v_clean}\n")
                                        f.write('\n')
                            if reply == '' and "publication_link_content" in paper and paper["publication_link_content"] and paper["publication_link_content"] != '':
                                with open(output_txt, "a", encoding="utf-8") as f:
                                    f.write(f"PUBLICATION WEBSITE of potential {exp_calc} paper: {paper['google_scholar_title']}: {paper['publication_link']}\n")
                                try:
                                    prompt = prepare_prompt_with_content(
                                        id = id,
                                        paper = paper,
                                        c = c,
                                        exp_calc = exp_calc,
                                        content = paper["publication_link_content"],
                                        llm_reviewer = llm_reviewer,
                                        rag = "keyword",
                                        fmt = "HTML",
                                        keywords = paper["keywords"],
                                        output_txt = output_txt,
                                    )
                                    if prompt == "":
                                        raise ValueError("No relevant content found in RAG.")
                                    reply = llm_reviewer.search_text(llm_reviewer.model2, exp_calc, prompt, getattr(llm_reviewer, f"second_prompt_{exp_calc}_schema"))
                                    reply_json = json.loads(reply)
                                except ValueError as e:
                                    print(f"RAG failed for paper {paper['google_scholar_title']}: {e}")
                                except json.JSONDecodeError as e:
                                    print(f"Could not parse JSON from model 2 reply: {reply}\nError: {e}")
                                else:
                                    with open(output_txt, "a", encoding="utf-8") as f:
                                        for k, v in reply_json.items():
                                            k_clean = re.sub("\n", "", str(k))
                                            v_clean = re.sub("\n", "", str(v))
                                            f.write(f"{k_clean}: {v_clean}\n")
                                        f.write('\n')
                with open(output_txt, "a", encoding="utf-8") as f:
                    f.write("--------------------------------\n")
                    f.write(f"{LLMReviewer.__name__} with model {llm_model} and keyword RAG is done LLM review for {id}...\n")

            for rag in rags:
                if rag == "keyword":
                    continue
                print(f"Using LLM model: gpt-5.2 with {rag} RAG...")
                llm_reviewer = LLMReviewer(id, output_txt, model2 = "gpt-5.2")
                with open(output_txt, "a", encoding="utf-8") as f:
                    f.write(f"{LLMReviewer.__name__} with model {llm_reviewer.model2} and {rag} RAG is doing LLM review for {id}...\n")
                    f.write("--------------------------------\n")
                
                for paper in tqdm(c.citing_papers[start_idx : start_idx + step], smoothing = 0.1, ncols = 80):
                    if paper["score"] > 0.7:
                        for exp_calc in ["exp", "calc"]:
                            if not set(paper["keywords"]) & set(getattr(keyword_reviewer, f"keywords_{exp_calc}")):
                                continue
                            reply = ''
                            if reply == '' and "pdf_link_content" in paper and paper["pdf_link_content"]:
                                with open(output_txt, "a", encoding="utf-8") as f:
                                    f.write(f"PDF of potential {exp_calc} paper: {paper['google_scholar_title']}: {paper['pdf_link']}\n")
                                try:
                                    prompt = prepare_prompt_with_content(
                                        id = id,
                                        paper = paper,
                                        c = c,
                                        exp_calc = exp_calc,
                                        content = paper['pdf_link_content'],
                                        llm_reviewer = llm_reviewer,
                                        rag = rag,
                                        fmt = "PDF",
                                        keywords = paper["keywords"],
                                        output_txt = output_txt,
                                    )
                                    if prompt == "":
                                        raise ValueError("No relevant content found in RAG.")
                                    reply = llm_reviewer.search_text(llm_reviewer.model2, exp_calc, prompt, llm_reviewer.second_prompt_exp_schema if exp_calc == "exp" else llm_reviewer.second_prompt_calc_schema)
                                    reply_json = json.loads(reply)
                                except ValueError as e:
                                    print(f"RAG failed for paper {paper['google_scholar_title']}: {e}")
                                except json.JSONDecodeError as e:
                                    print(f"Could not parse JSON from model 2 reply: {reply}\nError: {e}")
                                else:
                                    with open(output_txt, "a", encoding="utf-8") as f:
                                        for k, v in reply_json.items():
                                            k_clean = re.sub("\n", "", str(k))
                                            v_clean = re.sub("\n", "", str(v))
                                            f.write(f"{k_clean}: {v_clean}\n")
                                        f.write('\n')
                            if reply == '' and "publication_link_content" in paper and paper["publication_link_content"]:
                                with open(output_txt, "a", encoding="utf-8") as f:
                                    f.write(f"PUBLICATION WEBSITE of potential {exp_calc} paper: {paper['google_scholar_title']}: {paper['publication_link']}\n")
                                try:
                                    prompt = prepare_prompt_with_content(
                                        id = id,
                                        paper = paper,
                                        c = c,
                                        exp_calc = exp_calc,
                                        content = paper["publication_link_content"],
                                        llm_reviewer = llm_reviewer,
                                        rag = rag,
                                        fmt = "HTML",
                                        keywords = paper["keywords"],
                                        output_txt = output_txt,
                                    )
                                    if prompt == "":
                                        raise ValueError("No relevant content found in RAG.")
                                    reply = llm_reviewer.search_text(llm_reviewer.model2, exp_calc, prompt, llm_reviewer.second_prompt_exp_schema if exp_calc == "exp" else llm_reviewer.second_prompt_calc_schema)
                                    reply_json = json.loads(reply)
                                except ValueError as e:
                                    print(f"RAG failed for paper {paper['google_scholar_title']}: {e}")
                                except json.JSONDecodeError as e:
                                    print(f"Could not parse JSON from model 2 reply: {reply}\nError: {e}")
                                else:
                                    with open(output_txt, "a", encoding="utf-8") as f:
                                        for k, v in reply_json.items():
                                            k_clean = re.sub("\n", "", str(k))
                                            v_clean = re.sub("\n", "", str(v))
                                            f.write(f"{k_clean}: {v_clean}\n")
                                        f.write('\n')
                with open(output_txt, "a", encoding="utf-8") as f:
                    f.write("--------------------------------\n")
                    f.write(f"{LLMReviewer.__name__} with model gpt-5.2 and {rag} RAG is done LLM review for {id}...\n")
                    if rag == "keyword_embedding":
                        f.write(f"RAG stats: {llm_reviewer.rag_stats}\n")
                        print(llm_reviewer.rag_stats)
            
            start_idx += step
            if start_idx >= len(c.citing_papers):
                message = f"All done for {id}.\n"
                with open(output_txt, "a", encoding="utf-8") as f:
                    f.write(message)
                print(message)
            else:
                message = f"{start_idx}/{len(c.citing_papers)} papers done for {id}.\n"
                with open(output_txt, "a", encoding="utf-8") as f:
                    f.write(message)
                print(message)

        start_idx = 0
    
    driver.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total elapsed time: {hours:02d}h {minutes:02d}m {seconds:02d}s")

    return



if __name__ == "__main__":
    """
    service = Service("C:/Users/18000/OneDrive/Desktop/VSCode/chromedriver-win64/chromedriver.exe")
    driver = webdriver.Chrome(service=service)

    id = "ZZZDKE01"
    paper = {
                "publication_link": "https://www.cell.com/iscience/fulltext/S2589-0042(19)30332-3",
                "pdf_link": "https://www.cell.com/iscience/pdf/S2589-0042(19)30332-3.pdf",
                "google_scholar_title": "Anisotropic singlet fission in single crystalline hexacene",
                "google_scholar_snippet": "Singlet fission is known to improve solar energy utilization by circumventing the Shockley-Queisser limit. The two essential steps of singlet fission are the formation of a correlated …"
            }

    id = "QQQCIG13"
    paper = {
                "publication_link": "https://pubs.rsc.org/en/content/articlehtml/2010/jm/b914334c",
                "pdf_link": "https://pubs.rsc.org/en/content/articlepdf/2010/jm/b914334c",
                "google_scholar_title": "Rubrene micro-crystals from solution routes: their crystallography, morphology and optical properties",
                "google_scholar_snippet": "A series of rubrene micro-crystals (MCs) with controllable sizes and shapes, ranging from one-dimensional (1D) ribbons to 2D rhombic and hexagonal plates, have bee prepared by …"
            }

    id = "HBZCOR"
    paper = {
                "publication_link": "https://pubs.acs.org/doi/abs/10.1021/jp5064462",
                "pdf_link": "https://pubs.acs.org/doi/pdf/10.1021/jp5064462",
                "google_scholar_title": "High-throughput investigation of the geometry and electronic structures of gas-phase and crystalline polycyclic aromatic hydrocarbons",
                "google_scholar_snippet": "The quest for cheap, light, flexible materials for use in electronics applications has resulted in the exploration of soft organic materials as possible candidates, and several polycyclic …"
            }

    id = "KUBWAF01"
    paper = {
                "publication_link": "https://pubs.acs.org/doi/abs/10.1021/jacs.2c12574",
                "pdf_link": "https://pubs.acs.org/doi/pdf/10.1021/jacs.2c12574",
                "google_scholar_title": "Dibenzotropylium-capped orthogonal geometry enabling isolation and examination of a series of hydrocarbons with multiple 14π-aromatic units",
                "google_scholar_snippet": "A series of six dications composed of pure hydrocarbons with one to six non-substituted 9, 10-anthrylene units end-capped with two dibenzotropyliums were designed and synthesized …"
            }

    paper = {
                "publication_link": "https://chemistry-europe.onlinelibrary.wiley.com/doi/abs/10.1002/chem.202401063",
                "pdf_link": "",
                "google_scholar_title": "Overcrowded 14,14′‐Bidibenzo[a,j]anthracenes: Challenges in Syntheses and Atypical Property of Lacking Symmetry‐Breaking Charge Transfer (SBCT)",
                "google_scholar_snippet": "Bidibenzo [a, j] anthracenes (BDBAs) were prepared by iridium‐catalyzed annulation of 5, 5′‐biterphenylene with alkynes. The molecular geometries of overcrowded BDBAs were …"
            }

    id = "BEANTR"
    paper = {
                "publication_link": "https://www.degruyterbrill.com/document/doi/10.1515/zpch-1987-26849/html",
                "pdf_link": "",
                "google_scholar_title": "Vibronisches Spektralverhalten von Molekülen: VII. Theoretische Untersuchungen zur Molekülgeometrie und Feinstruktur der Emissionsspektren des 1, 2 …",
                "google_scholar_snippet": "The theoretical completely-optimized molecular geometries of 1, 2-benzanthracene are presented for the electronic states S0, Sit S¡ and 7\\, being of interest to the spectroscopy of …"
            }

    id = "BNPERY"
    paper = {
                "publication_link": "https://pubs.rsc.org/en/content/articlehtml/2018/qm/c8qm00112j",
                "pdf_link": "https://pubs.rsc.org/en/content/getauthorversionpdf/c8qm00112j",
                "google_scholar_title": "Charge-transfer complexes based on C 2v-symmetric benzo [ghi] perylene: comparison of their dynamic and electronic properties with those of D 6h-symmetric …",
                "google_scholar_snippet": "Single crystals of three neutral charge-transfer complexes and a cation radical salt based on a C2v-symmetric polycyclic aromatic hydrocarbon, benzo [ghi] perylene (bper), were …"
            }

    paper = {
                "publication_link": "https://iopscience.iop.org/article/10.1088/0370-1328/83/1/312/meta",
                "pdf_link": "",
                "google_scholar_title": "Anisotropy of the electronic spectra of a single crystal of 1, 12-benzperylene (C22H12)",
                "google_scholar_snippet": "The polarized fluorescence and absorption spectra of a 1, 12-benzperylene crystal have been measured at 300 K. The intensity ratios measured along the b and a axes are found to …"
            }
    paper = {
                "publication_link": "https://www.collectionscanada.gc.ca/obj/thesescanada/vol2/002/NR44349.PDF?is_thesis=1&oclc_number=669240940",
                "pdf_link": "https://www.collectionscanada.gc.ca/obj/thesescanada/vol2/002/NR44349.PDF?is_thesis=1&oclc_number=669240940",
                "google_scholar_title": "[BOOK][B] Isoelectric Analogues of Polycyclic Aromatic Hydrocarbons Incorporating Boron and Nitrogen",
                "google_scholar_snippet": "The B= N moiety is isosteric and isoelectronic with the C= C bond, and as such substitution of this unit in polycyclic aromatic hydrocarbons (PAHs) may be expected to have a minimal …"
            }

    id = "BIPHEN"
    paper = {
                "publication_link": "https://www.sciencedirect.com/science/article/pii/S0379677900004318",
                "pdf_link": "https://www.sciencedirect.com/science/article/pii/S0379677900004318",
                "google_scholar_title": "[HTML][HTML] Pressure studies on the intermolecular interactions in biphenyl",
                "google_scholar_snippet": "We investigate the influence of intermolecular interactions on the optical and structural properties of oligophenyls in solid films of polycrystalline nature. To this end, we have …"
            }

    keyword_llm_paper(
        id, 
        paper, 
        driver, 
        rag = "keyword",
        )
    driver.close()
    """
    # pah101_pipeline()
    # pah101_pipeline(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    # pah101_paper()

    # folder = "kaiji"
    folder = "sub1ev"
    # folder = "100-125ev"
    # folder = "125-150ev"
    # folder = "150-175ev"
    # folder = "175-200ev"
    ids = [i.split('.')[0] for i in os.listdir(f"json/{folder}/")]
    keyword_llm_pipeline(ids, f"LiteratureReview/{folder}.txt")

