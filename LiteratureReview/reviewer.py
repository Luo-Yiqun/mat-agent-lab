"""
@ Yiqun Luo (luo2@andrew.cmu.edu)

This script defines review classes for literature review tasks.

Please run the code under the campus network for access to databases.
The current keyword search version needs manually extracting data, an LLM would further automate that.

STRONGLY recommend to use selenium to avoid access forbidden (403) error, although it cannot be avoided sometimes.
If you want to use selenium, please download and install the chromedriver and set the path to the chromedriver.
You may need to manually answer robot verification, especially for logining to Google Scholar for the first time.

Please put the config.json containing the OpenAI API under the LiteratureReview dataset!
"""



import time
import random
import json
from typing import Any, Literal, Optional
import io
import base64
import re
import numpy as np
import math
import scipy

from tqdm import tqdm

import requests
from pypdf import PdfReader
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from curl_cffi import requests as cfre

try:
    import openai
except ImportError as e:
    print(e, "No OpenAI Python API available!")
from sentence_transformers import SentenceTransformer

from utilities import *
from html_parser import RAGFlowHtmlParser



def normal_seperation(mu1, sigma1, mu2, sigma2):
    if sigma1 <= 0 or sigma2 <= 0:
        return None
    elif abs(sigma1 - sigma2) < 1e-12:
        return (mu1 + mu2) / 2
    # solve quadratic for intersection(s)
    a = 1/(2*sigma1**2) - 1/(2*sigma2**2)
    b = mu2/(sigma2**2) - mu1/(sigma1**2)
    c = mu1**2/(2*sigma1**2) - mu2**2/(2*sigma2**2) - math.log(sigma2/sigma1)
    disc = b**2 - 4*a*c
    if disc <= 0:  # practically identical or disjoint
        return min(1.0, max(0.0, math.sqrt(sigma1*sigma2) / (
            ((sigma1**2 + sigma2**2)/2)**0.5) *
            scipy.stats.norm.cdf(-(mu1 - mu2) / math.sqrt(sigma1**2 + sigma2**2))))
    x1 = (-b - math.sqrt(disc)) / (2*a)
    x2 = (-b + math.sqrt(disc)) / (2*a)
    # choose the intersection that lies between the means
    if min(mu1, mu2) <= x1 <= max(mu1, mu2):
        x = x1
    elif min(mu1, mu2) <= x2 <= max(mu1, mu2):
        x = x2
    else:
        x = None
    return x



class Reviewer:
    """
    A base class to review the citing papers using certain methods.
    """
    @staticmethod
    def str_contains_keyword(text: str, keywords: list[str]) -> list[str]:
        """
        Check if the text contains any of the keywords.
        """
        return [keyword for keyword in keywords if re.sub(r'\s+', '', keyword).lower() in re.sub(r'\s+', '', text).lower()]


    def __init__(self, id: str = None, output_file: str = "output.txt", driver: webdriver.Chrome = None):
        self.id = id
        self.output_file = output_file
        if driver:
            self.driver = driver


    def search(self, papers: list[dict]) -> list:
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write(f"{self.__class__.__name__} is searching results for {self.id}...\n")
            f.write("--------------------------------\n")
        n_scrape_success = 0
        n_hit = 0
        for paper in tqdm(papers, smoothing = 0.1, ncols = 80):
            self.search_one(paper)
            if paper["scrape_success"]:
                n_scrape_success += 1
            else:
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(f"Warning: Scrape failed for paper: {paper.get('google_scholar_title', '')}\n")
            if paper["score"] > 0.7:
                n_hit += 1
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write("--------------------------------\n")
            f.write(f"{n_scrape_success}/{len(papers)} papers scraped successfully.\n")
            if n_hit == 0:
                if len(papers):
                    f.write(f"Citing papers found but no result found for {self.id}.\n")
                else:
                    f.write(f"Citing papers not found successfully.\n")
        return



class KeyWordSearch(Reviewer):
    """
    A class to review the citing papers by doing a keyword search to find certain properties.
    """
    def __init__(
        self,
        id: str = None,
        output_file: str = "output.txt",
        keywords_exp: list[str] = None,
        keywords_calc: list[str] = None,
        keywords_other: list[str] = ["singlet fission", "triplet-triplet annihilation", "thermally activated delayed fluorescence", "TADF"],
        driver: webdriver.Chrome = None,
        use_scihub: bool = True
        ):
        super().__init__(id, output_file, driver)
        # I did not include the abbreviations GW, BSE, PES and UPS here because there are too many false positives.
        if keywords_exp:
            self.keywords_exp = keywords_exp
        else:
            self.keywords_exp = ["photoemission spectroscopy", "IPES", "inverse photoemission spectroscopy",
                                 "ultraviolet photoemission spectroscopy", "2PPE", "two-photon photoemission",
                                 "optical gap", "band gap", "optical band gap", "singlet excitation energy",
                                 "absorption", "UV-Vis",
                                 "fluorescence", "photoluminescence", "fluorescent"]
        if keywords_calc:
            self.keywords_calc = keywords_calc
        else:
            self.keywords_calc = ["Green's function", "GW approximation", "Bethe-Salpeter Equation"]
        if keywords_other:
            self.keywords_other = keywords_other
        else:
            self.keywords_other = []
        self.keywords = self.keywords_exp + self.keywords_calc + self.keywords_other

        self.use_scihub = use_scihub


    def url_to_html(self, url: str) -> str:
        """
        Convert the url content to text.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        try:
            if self.driver:
                self.driver.get(url)
                try:
                    WebDriverWait(self.driver, 20)
                except Exception:
                    return ''
                else:
                    text = self.driver.page_source
            else:
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 403:
                    print(f"Access forbidden (403) for {url}. Will retry...")
                    time.sleep(random.uniform(5, 10))
                    resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    print(f"HTTP {resp.status_code} for {url}. Skipping...")
                    return ""
                text = resp.text
            
            return text
        except Exception as e:
            print(f"Unexpected error fetching {url}: {e}")
            return ""
    

    def paper_to_pdf(self, paper: dict) -> str:
        """
        Convert the pdf link content to text.
        """
        if self.use_scihub:
            pdf_link = paper.get("pdf_link", "")
            if self.driver:
                """
                Sci-Hub provides multiple websites.
                After test,
                sci-hub.su: has access issue for sciencedirect link sometimes,
                sci-hub.se/st: works well for ACS and sciencedirect links.
                """
                suffixes = ("ru", "st", "su", "box", "red")
                # suffixes = ("ru", "se", "st", "su", "box", "red")
                for key in ("doi", "publication_link", "google_scholar_title"):
                    if key in paper and paper[key] != "" and "..." not in paper[key]:
                        time.sleep(random.uniform(1, 3))
                        try:
                            # The get statement may fail sometimes for unknown reasons.
                            url = "https://sci-hub." + random.choice(suffixes) + "/"
                            self.driver.get(url)
                            WebDriverWait(self.driver, 20)
                            form_el = self.driver.find_element(By.TAG_NAME, "form")
                        except Exception:
                            # Restart the selenium driver if Sci-Hub page load / wait fails
                            _service_path = getattr(getattr(self.driver, "service", None), "path", None)
                            _options = getattr(self.driver, "options", None)
                            self.driver.quit()
                            time.sleep(random.uniform(1, 3))
                            self.driver = webdriver.Chrome(service=Service(_service_path), options=_options)
                            try:
                                url = "https://sci-hub." + random.choice(suffixes) + "/"
                                self.driver.get(url)
                                WebDriverWait(self.driver, 20)
                                form_el = self.driver.find_element(By.TAG_NAME, "form")
                            except Exception:
                                print(f"Error accessing {url} for {key} of paper {paper.get('google_scholar_title', '')}. Skipping Sci-Hub access for this key.")
                                break
                        try:
                            text = paper[key]
                            if key == "google_scholar_title":
                                # Google Scholar titles sometimes include bracketed tags like "[PDF]" or "[HTML]".
                                # Remove any square-bracketed segments and normalize whitespace.
                                text = re.sub(r"\[[^\]]*\]", "", text)
                                text = re.sub(r"\s+", " ", text).strip()
                            form_el.find_element(By.TAG_NAME, "textarea").send_keys(text)
                            form_el.find_element(By.CSS_SELECTOR, "button, input[type='submit']").click()
                            WebDriverWait(self.driver, 20)
                            try:
                                gate = self.driver.find_element(By.XPATH, "/html/body/div[@class='question']/div[@class='answer']")
                            except Exception:
                                gate = None
                            else:
                                if gate:
                                    time.sleep(random.uniform(1, 3))
                                    gate.click()
                            pdf_link = self.driver.find_element(By.XPATH, "/html/body/div[@class='menu']/div[@class='panel']/div[@class='download']/a").get_property("href")
                            WebDriverWait(self.driver, 20)
                        except Exception:
                            continue
                        else:
                            break
            else:
                print("Warning: Sci-Hub access requires selenium driver. Skipping Sci-Hub.")
        else:
            pdf_link = paper.get("pdf_link", "")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Referer": pdf_link,
        }

        try:
            resp = cfre.get(
                pdf_link,
                headers=headers,
                impersonate="chrome",     # important: uses Chrome-like TLS fingerprint
                allow_redirects=True,
                timeout=120,
            )
            # resp = requests.get(pdf_url, headers=headers, timeout=20)
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code} for PDF {pdf_link}. Skipping...")
                return ""
            pdf_bytes = io.BytesIO(resp.content)
            try:
                reader = PdfReader(pdf_bytes)
                text = ""
                for page in reader.pages:
                    try:
                        text += page.extract_text() or ""
                    except Exception as e:
                        print(f"Error extracting text from a page in {pdf_link}: {e}")
                        return ""
                return text
            except Exception as e:
                print(f"Error reading PDF from {pdf_link}: {e}")
                return ""
        except Exception as e:
            print(f"Unexpected error fetching PDF {pdf_link}: {e}")
            return ""


    def html_contains_keyword(self, url: str) -> tuple[list[str], str]:
        """
        Check if the url contains any of the keywords.
        """
        text = self.url_to_html(url)
        if text == "":
            return [], ""
        else:
            return Reviewer.str_contains_keyword(text, self.keywords), text


    def pdf_contains_keyword(self, paper: dict) -> tuple[list[str], str]:
        """
        Check if the pdf link contains any of the keywords.
        """
        text = self.paper_to_pdf(paper)
        if text == "":
            return [], ""
        else:
            return Reviewer.str_contains_keyword(text, self.keywords), text


    def search_one(self, paper: dict) -> dict:
        """
        Review one paper.
        """
        if (paper == {}):
            print(f"Warning: Paper is empty. Skipping...")
            return 0
        paper["scrape_success"] = False
        paper["score"] = 0
        paper["keywords"] = set()

        # Search in the Google Scholar title
        keywords = Reviewer.str_contains_keyword(paper.get("google_scholar_title", ""), self.keywords)
        if len(keywords) > 0:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"Keyword {keywords} found in the TITLE of paper: {paper.get('google_scholar_title', '')}\n")
            paper["keywords"].update(keywords)
            paper["score"] = 1

        # Search in the Google Scholar snippet
        keywords = Reviewer.str_contains_keyword(paper.get("google_scholar_snippet", ""), self.keywords)
        if len(keywords) > 0:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"Keyword {keywords} found in the SNIPPET of paper: {paper.get('google_scholar_title', '')}:\n{paper.get('google_scholar_snippet', '')}\n")
            paper["keywords"].update(keywords)
            paper["score"] = 1
        
        # Search in the publication HTML
        if "publication_link" in paper and paper["publication_link"]:
            keywords, content = self.html_contains_keyword(paper["publication_link"])
            if content != "":
                paper["scrape_success"] = True
            if len(keywords) > 0:
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(f"Keyword {keywords} found in the PUBLICATION WEBSITE for paper: {paper.get('google_scholar_title', '')}: {paper.get('publication_link', '')}\n")
                paper["keywords"].update(keywords)
                paper["publication_link_content"] = content
                paper["score"] = 1
        
        # Search in the PDF
        keywords, content = self.pdf_contains_keyword(paper)
        if content != "":
            paper["scrape_success"] = True
        if len(keywords) > 0:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"Keyword {keywords} found in the PDF for paper: {paper.get('google_scholar_title', '')}: {paper.get('pdf_link', '')}\n")
            paper["keywords"].update(keywords)
            paper["pdf_link_content"] = content
            paper["score"] = 1
            
        return paper



class LLMReviewer(Reviewer):
    """
    A class to review the citing papers using LLM.
    """
    # Optimized prompts for LLMReviewer
    static_prompt_exp = (
        "You are the legacy literature-review sub-agent inside a broader multi-agent materials workflow. "
        "Your job here is paper-grounded screening and extraction for a target polymorph identified by CSD reference code, SMILES, or related context. "
        "You may support downstream tasks such as cited-paper discovery, question answering, and simulation preparation, but in this legacy step you must only judge and extract evidence from the current paper. "
        "Given a paper (text only), assess if it meets ALL of these criteria:\n"
        "1. The paper reports experimental (not theoretical or computational) work on a specific polymorph.\n"
        "2. The material is in an allowed phase:\n"
        "   - Allowed: Gas phase; Solid phase (single crystal, polycrystal, thin film, bulk, etc.)\n"
        "   - Not allowed: Solution phase\n"
        "3. The paper discusses at least one of these properties (and only these):\n"
        "   - Density of states (DOS): photoemission spectroscopy, IPES, inverse photoemission spectroscopy, ultraviolet photoemission spectroscopy, 2PPE, two-photon photoemission spectroscopy\n"
        "   - Absorption spectrum: absorption spectrum, UV-Vis, fluorescence, photoluminescence, fluorescent\n"
        "   - Optical gap (including terms such as optical gap, band gap, or optical band gap)\n"
        "\n"
        "Assumption rule:\n"
        "- If there is not enough crystal information, such as the space group or lattice constant, is provided, but the experiment clearly concerns the same molecule as the target (by SMILES/name), assume the polymorph is the one of interest.\n"
        "\n"
        "Instructions:\n"
        "- Use only the information provided. Do not assume beyond the text except for the Assumption rule above.\n"
        "- Treat this as a cited-paper evidence extraction task, not an open-ended chat task.\n"
        "- Prefer explicit, auditable evidence that can later support grounded answers in the larger workflow.\n"
        "- If information is missing or ambiguous, do not guess (other than the Assumption rule).\n"
        "- Be concise and objective in your evaluation."
    )
    first_prompt_exp = (
        "Based on the provided paper information and the criteria above, "
        "estimate a confidence score between 0 (does not satisfy) and 1 (fully satisfies) "
        "for whether the paper meets ALL of the following:\n"
        "1. It is an experimental paper about the given polymorph.\n"
        "2. The material is in an allowed phase (not solution).\n"
        "3. The paper discusses at least one of the relevant properties.\n"
        "This score is used to decide whether the paper should continue in the cited-paper extraction workflow.\n"
        "Respond with only the confidence score as a number between 0 and 1."
    )
    first_prompt_exp_schema = {
        "type": "number",
        "minimum": 0,
        "maximum": 1
    }
    second_prompt_exp = (
        "The previous confidence score exceeded the threshold.\n"
        "Now, extract the following experimental results for the polymorph of interest so they can support downstream question answering and reporting, and return a STRICT JSON object with EXACTLY these keys:\n"
        "['confidence_score', 'density_of_state', 'absorption_spectrum', 'optical_gap', 'optical_gap_source']\n"
        "Instructions:\n"
        "- Use ONLY the provided material. Do NOT include theoretical or computational results. Do NOT infer or guess beyond the text.\n"
        "- For 'density_of_state' and 'absorption_spectrum': return ONLY a figure/table/scheme reference (e.g., 'Fig. 3', 'Figure 2a', 'Fig. S4', 'Table 1'). "
        "  If the paper mentions the technique (e.g., 'absorption spectroscopy') but does NOT provide a specific figure/table/scheme reference, return null.\n"
        "- If a field is not explicitly mentioned, return null (the JSON literal).\n"
        "- Only include results for the specified polymorph; ignore other polymorphs.\n"
        "- Prefer evidence that is precise enough to be cited later in a final answer.\n"
        "- Output STRICTLY valid JSON (double-quoted keys/strings, no trailing commas), and NOTHING else."
    )
    second_prompt_exp_schema = {
        "type": "object",
        "properties": {
            "confidence_score": {
                "type": "number",
                "description": "A confidence score between 0 and 1",
                "minimum": 0,
                "maximum": 1
            },
            "density_of_state": {
                "type": ["string", "null"],
                "description": "figure/table/scheme reference (e.g., \"Fig. 2\", \"Figure 3a\", \"Table 1\", \"Fig. S4\"), or null if not present",
                "maxLength": 64,
                "pattern": "^(Fig(?:ure)?|Table|Tab|Scheme)\\.?\\s*(?:S\\s*)?\\d+[A-Za-z]?(?:\\s*\\([A-Za-z0-9]+\\))?(?:\\s*[-–]\\s*(?:S\\s*)?\\d+[A-Za-z]?)?$"
            },
            "absorption_spectrum": {
                "type": ["string", "null"],
                "description": "figure/table/scheme reference (e.g., \"Fig. 3\", \"Figure 2a\", \"Table 2\", \"Fig. S1\"), or null if not present",
                "maxLength": 64,
                "pattern": "^(Fig(?:ure)?|Table|Tab|Scheme)\\.?\\s*(?:S\\s*)?\\d+[A-Za-z]?(?:\\s*\\([A-Za-z0-9]+\\))?(?:\\s*[-–]\\s*(?:S\\s*)?\\d+[A-Za-z]?)?$"
            },
            "optical_gap": {
                "type": ["number", "null"],
                "description": "number in eV (no units), or null if not present"
            },
            "optical_gap_source": {
                "type": ["string", "null"],
                "description": "short quote (<=200 characters) or figure/table reference supporting optical_gap, or null if not present"
            },
        },
        "required": ["confidence_score", "density_of_state", "absorption_spectrum", "optical_gap", "optical_gap_source"],
        "additionalProperties": False
    }

    static_prompt_calc = (
        "You are the legacy literature-review sub-agent inside a broader multi-agent materials workflow. "
        "Your role here is paper-grounded screening and extraction for a target polymorph identified by CSD reference code, SMILES, or related context. "
        "You may support downstream tasks such as cited-paper discovery, question answering, and simulation preparation, but in this legacy step you must only judge and extract computational evidence from the current paper. "
        "Given a paper (text only), assess if it meets ALL of these criteria:\n"
        "1. The paper reports at least one of the following computational results (not experimental):\n"
        "   - GW approximation (GWA)\n"
        "   - Bethe-Salpeter Equation (BSE)\n"
        "2. The paper concerns a specific polymorph (not just a molecule).\n"
        "\n"
        "Assumption rule:\n"
        "- If there is not enough crystal information, such as the space group or lattice constant, is provided, but the study clearly concerns the same molecule as the target (by SMILES/name), assume the polymorph is the one of interest.\n"
        "\n"
        "Instructions:\n"
        "- Use only the provided information. Do not infer beyond the text except for the Assumption rule above.\n"
        "- Treat this as a cited-paper evidence extraction task, not an open-ended chat task.\n"
        "- Prefer explicit, auditable computational evidence that can later support grounded answers in the larger workflow.\n"
        "- If information is missing or ambiguous, do not guess (other than the Assumption rule).\n"
        "- Be concise and objective."
    )
    first_prompt_calc = (
        "Based on the provided paper information and the criteria above, "
        "estimate a confidence score between 0 (does not satisfy) and 1 (fully satisfies) "
        "for whether the paper meets ALL of the following:\n"
        "1. It is a computational paper that reports at least one of the computational results (not experimental):\n"
        "2. The paper is on the specific polymorph (not molecule).\n"
        "This score is used to decide whether the paper should continue in the cited-paper extraction workflow.\n"
        "Respond with only the confidence score as a number between 0 and 1."
    )
    first_prompt_calc_schema = {
        "type": "number",
        "minimum": 0,
        "maximum": 1
    }
    second_prompt_calc = (
        "The previous confidence score exceeded the threshold. "
        "Now, extract the following computational results for the polymorph of interest so they can support downstream question answering and reporting, and return a STRICT JSON object with EXACTLY these keys:\n"
        "['confidence_score', 'bse_gap', 'bse_gap_source', 'other_gw_bse_results']\n"
        "Instructions:\n"
        "- Include ONLY computational GW/BSE evidence; EXCLUDE experimental results. Do not infer or guess beyond the text.\n"
        "- Consider ONLY the specified polymorph; ignore other polymorphs.\n"
        "- If a field is not explicitly supported by the text, use null (or [] for the list).\n"
        "- Prefer evidence that is precise enough to be cited later in a final answer.\n"
        "- Output STRICTLY valid JSON (double-quoted keys/strings, no trailing commas), and NOTHING else."
    )
    second_prompt_calc_schema = {
        "type": "object",
        "properties": {
            "confidence_score": {
                "type": "number",
                "description": "A confidence score between 0 and 1",
                "minimum": 0,
                "maximum": 1
            },
            "bse_gap": {
                "type": ["number", "null"],
                "description": "number in eV (no units), or null if not present"
            },
            "bse_gap_source": {
                "type": ["string", "null"],
                "description": "short quote (<=200 characters) or figure/table reference supporting bse_gap, or null if not present"
            },
            "other_gw_bse_results": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "short quote (<=200 characters) or figure/table reference on other GW/BSE results"
                },
                "description": "list of short quotes (<=200 characters each) or figure/table references on other GW/BSE results, or an empty list if not present"
            },
        },
        "required": ["confidence_score", "bse_gap", "bse_gap_source", "other_gw_bse_results"],
        "additionalProperties": False
    }

    # Jacqueline Cole
    # Search SI
    def __init__(
        self, 
        id: str = None,
        output_file: str = "output.txt", 
        llm_log_file: str = "llm_log.txt",
        openai_api_key: str = None, 
        model1: str = "gpt-5-nano", 
        model2: str = "gpt-5-nano",
        threshold: float = 0,
        temperature: float = 0.2,
        driver: webdriver.Chrome = None,
        ):
        super().__init__(id, output_file, driver)
        self.llm_log_file = llm_log_file
        self.model1 = model1
        self.model2 = model2
        self.threshold = threshold
        self.temperature = temperature

        if openai_api_key:
            self.openai_api_key = openai_api_key
            self.base_url = None
        else:
            with open("config.json", "r", encoding="utf-8") as f:
                key = json.load(f)
                self.openai_api_key = key["api_key"]
                self.base_url = key.get("base_url")
        if openai_api_key and not hasattr(self, "base_url"):
            self.base_url = None
        self.client = None
        if self.openai_api_key:
            self.client = openai.OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
        
        self.rag_stats = {
            "keyword_mean": [],
            "keyword_std": [],
            "embedding_mean": [],
            "embedding_std": [],
            "embedding_high_mean": [],
            "embedding_high_std": [],
            "embedding_low_mean": [],
            "embedding_low_std": [],
            "embedding_separation": [],
        }


    # Cache for locally loaded embedding backends (Specter2, MatSciBERT, etc.)
    _embedder_cache: dict[str, Any] = {
        "m3rg-iitd/matscibert": SentenceTransformer("m3rg-iitd/matscibert"),
    }
    _rag_embedding_params = {
        "openai": {
            "keyword_weight": 0.9,
            "embedding_weight": 0.1,
            "similarity_threshold": 0.644, # (0.632)
        },
        "matscibert": {
            "model_name": "m3rg-iitd/matscibert",
            "keyword_weight": 0.05,
            "embedding_weight": 0.95,
            "similarity_threshold": 0.743 
        },
    }

    @classmethod
    def _get_embedder(cls, model_name: str):
        """Load and cache a SentenceTransformer model lazily."""
        if model_name not in cls._embedder_cache:
            cls._embedder_cache[model_name] = SentenceTransformer(model_name)
        return cls._embedder_cache[model_name]


    # --- Lightweight tokenizer utilities (abandon external rag_tokenizer) ---
    @staticmethod
    def _tok_list(text: str) -> list[str]:
        """Regex-based, corpus-free tokenization (lowercased alphanumerics)."""
        return re.findall(r"[A-Za-z0-9]+", text.lower())

    @staticmethod
    def _tok_count(text: str) -> int:
        return len(LLMReviewer._tok_list(text))

    @staticmethod
    def _tok_spans(text: str) -> list[tuple[int, int]]:
        """Character spans for tokens over ORIGINAL text, aligned with _tok_list."""
        return [m.span() for m in re.finditer(r"[A-Za-z0-9]+", text)]

    @staticmethod
    def span_to_sentence(
        text: str,
        start_idx: int,
        end_idx: int,
        pre_chars: int = 800,
        post_chars: int = 1200,
        delims: str = ".!?。？！",
        include_newline: bool = False,
    ) -> tuple[int, int]:
        """
        Expand [start_idx, end_idx) to nearest sentence boundaries within a bounded window.

        - pre_chars/post_chars bound how far we search to the left/right.
        - delims lists sentence-ending characters to consider.
        - include_newline also treats "\n" as a boundary.
        Returns a half-open interval [start, end) clamped to text length and trimmed for whitespace.
        """
        n = len(text)
        if n == 0:
            return 0, 0
        # Clamp incoming indices
        start_idx = max(0, min(start_idx, n))
        end_idx = max(start_idx, min(end_idx, n))

        # Define a search window to avoid scanning entire document
        win_start = max(0, start_idx - max(0, pre_chars))
        win_end = min(n, end_idx + max(0, post_chars))

        # Left boundary: last delimiter before start_idx within the window
        left_region = text[win_start:start_idx]
        left_candidates = [left_region.rfind(d) for d in delims]
        if include_newline:
            left_candidates.append(left_region.rfind("\n"))
        left_pos = max([p for p in left_candidates if p != -1], default=-1)
        new_start = (win_start + left_pos + 1) if left_pos != -1 else win_start

        # Right boundary: first delimiter after end_idx within the window
        right_region = text[end_idx:win_end]
        right_candidates = [right_region.find(d) for d in delims]
        if include_newline:
            right_candidates.append(right_region.find("\n"))
        right_offsets = [p for p in right_candidates if p != -1]
        new_end = (end_idx + min(right_offsets) + 1) if right_offsets else win_end

        # Trim outer whitespace
        while new_start < new_end and new_start < n and text[new_start].isspace():
            new_start += 1
        while new_end > new_start and text[new_end - 1].isspace():
            new_end -= 1

        # Final clamp/sanity
        new_start = max(0, min(new_start, n))
        new_end = max(new_start, min(new_end, n))
        return new_start, new_end


    def rag_keyword_only(self, text: str, keywords: set[str], map_back: bool = True) -> list[str]:
        """
        Retrieve relevant information from the text using keywords.
        """
        if text == "" or text is None:
            return [""]
        
        if map_back:
            norm_text = []
            norm_to_orig: list[int] = []
            for i, ch in enumerate(text):
                if ch.isspace():
                    continue
                ch = ch.lower()
                norm_text.append(ch)
                norm_to_orig += [i] * len(ch)
            norm_text = "".join(norm_text)

            # Find all keyword occurrences in normalized text
            ret_idx = []
            for keyword in keywords:
                norm_keyword = re.sub(r'\s+', '', keyword).lower()
                index = norm_text.find(norm_keyword)
                while index != -1:
                    ret_idx.append(LLMReviewer.span_to_sentence(text, norm_to_orig[max(0, index - 800)], norm_to_orig[min(len(norm_text), index + 1200) - 1] + 1))
                    index = norm_text.find(norm_keyword, index + 1)

        else:
            # Simple normalization without mapping
            norm_text = re.sub(r'\s+', '', text)
            norm_text = re.sub(r'[\u200B\u200C\u200D\u2060\u00AD]', '', norm_text)
            norm_text = re.sub(r'@@[\t0-9 .\-\r\n]{0,256}##', '', norm_text)
            norm_text = norm_text.lower()

            ret_idx = []
            for keyword in keywords:
                keyword_norm = ''.join(keyword.split()).lower()
                index = norm_text.find(keyword_norm)
                while index != -1:
                    ret_idx.append(LLMReviewer.span_to_sentence(text, max(0, index - 800), min(len(text), index + 1200)))
                    index = norm_text.find(keyword_norm, index + 1)
        
        if not ret_idx:
            return [""]
        
        # Merge overlapping ranges and return combined text snippets
        ret_idx.sort()
        merged_ranges = []
        current_start, current_end = ret_idx[0]
        for start, end in ret_idx[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_ranges.append((current_start, current_end))
                current_start, current_end = start, end
        merged_ranges.append((current_start, current_end))

        rag_snippets = [text[start:end] for start, end in merged_ranges]
        return rag_snippets


    def rag_keyword_and_embedding(
        self,
        text: str,
        format: Literal["PDF", "HTML"],
        keywords: set[str],
        query: Optional[str] = None,
        chunk_token_num: int = 512,
        chunk_overlap: int = 0,
        embedding_backend: Literal["openai", "matscibert"] = "matscibert",
        emb_model: str = "text-embedding-3-large",
        top_k: int = 3,
        max_chunks: int = 512,
        prefilter_by_keywords: bool = True,
        verbose: bool = False,
        ) -> str:
        """
        Retrieve relevant information from the text using keywords and embeddings.

        PDF handling note:
        - PDFs often lack reliable blank-line paragraph separators after text extraction. We therefore
            chunk by lines, packing lines into chunks up to a token budget (chunk_token_num).
        - Optionally, you can set chunk_overlap (in number of TOKENS) to overlap the tail portion of a
            chunk into the next chunk for better cross-boundary recall. Overlap is computed from the
            end of the finalized chunk by taking as many trailing lines as needed to cover at least
            chunk_overlap tokens. Default is 0 (no overlap).
            
        Scoring note:
        - Keyword score is the count of distinct keywords in a chunk (not normalized). This typically differs by ~1
            between an irrelevant and a relevant chunk.
        - Embedding cosine similarities are mapped linearly from [-1,1] to [0,1] without per-batch min/max; this can
            differ by up to ~1 between irrelevant and relevant chunks.
        - Choose keyword_weight, embedding_weight, and similarity_threshold accordingly for your use case.

        Selection and limits:
        - If no chunk meets similarity_threshold, the top_k chunks by combined score are returned.
        - To control API cost/latency, at most max_chunks are embedded; prefilter_by_keywords keeps chunks containing
            any keyword before trimming.

        Embedding backends:
        - "openai": Use OpenAI text-embedding-3-* models via the API (default).
        - "specter2": Use the locally hosted allenai/specter2 SentenceTransformer model (requires sentence-transformers).
        - "matscibert": Use a materials-science tuned SentenceTransformer (default: m3hrdadfi/matscibert-uncased).

        (threshold, std, nomalized dev):
        Keyword: (0.5, 0.5), (0.5, 0.75), (0.5, 0.807)
        text-embedding-3-large:
        query similarity: (0.703, 0.055, 0.855), (?0.737, 0.012, -0.827), (0.763, 0.015, 0.300)
        -> keyword similarity: (0.662, 0.046, 1.157), (0.633, 0.042, 2.907), (0.688, 0.044, 4.027)
        matscibert:
        -> query similarity: (0.877, 0.048, 0.990), (0.904, 0.038, 1.590), (0.908, 0.027, 0.242)
        keyword similarity: (0.753, 0.036, 1.103), (0.741, 0.026, 1.159), (0.770, 0.018, 0.619)
        """
        # 0) Validate inputs and set defaults
        if not text or not isinstance(text, str):
            return ""
        if isinstance(keywords, set):
            keywords = list(keywords)
        if (not isinstance(keywords, list)) or len(keywords) == 0:
            print("Warning: keywords should be a list of strings. Setting to empty list is equivalent to neglecting keyword matching.")
        keywords = keywords or []
        # Build a fallback query from keywords if not provided
        if not query:
            print("Warning: No query provided; building query from keywords.")
            query = " ".join(keywords) if keywords else ""
        keyword_weight = LLMReviewer._rag_embedding_params[embedding_backend]["keyword_weight"]
        embedding_weight = LLMReviewer._rag_embedding_params[embedding_backend]["embedding_weight"]
        similarity_threshold = LLMReviewer._rag_embedding_params[embedding_backend]["similarity_threshold"]

        # I) Parsing and chunking
        chunks: list[str] = []
        try:
            if format == "HTML":
                # I-i) Use HTML parser to create token-aware chunks
                chunks = RAGFlowHtmlParser.parser_txt(text, chunk_token_num=chunk_token_num)
            elif format == "PDF":
                # I-ii) PDF: chunk by lines (paragraphs are unreliable in many extractions)
                # Todo: Whether to split based on delimiters
                lines = [ln.strip() for ln in text.splitlines()]
                current_lines: list[str] = []
                current_counts: list[int] = []  # per-line token counts to enable overlap
                current_tokens = 0

                for ln in lines:
                    if not ln.strip():
                        continue
                    # Tokenize for counting only; always output original substrings
                    t_count = LLMReviewer._tok_count(ln)

                    if t_count >= chunk_token_num:
                        # Current line itself too long: flush current chunk, then split this long line
                        # Although I think it might not happen often in practice
                        if current_lines:
                            chunks.append("\n".join(current_lines))
                        # Build token spans over the original line so we can slice original substrings
                        spans = LLMReviewer._tok_spans(ln)
                        total_tokens = len(spans)
                        overlap_tokens = max(0, min(chunk_overlap, chunk_token_num - 1))
                        start_tok = 0
                        while start_tok < total_tokens:
                            end_tok = min(start_tok + chunk_token_num, total_tokens)
                            if start_tok < end_tok:
                                start_char = spans[start_tok][0]
                                end_char = spans[end_tok - 1][1]
                                chunks.append(ln[start_char:end_char])
                            if end_tok >= total_tokens:
                                break
                            start_tok = end_tok - overlap_tokens
                        current_lines = []
                        current_counts = []
                        current_tokens = 0

                    else:
                        if current_tokens + t_count > chunk_token_num:
                            # finalize current chunk
                            if current_lines:
                                chunks.append("\n".join(current_lines))
                            # prepare next chunk with optional overlap (by tokens)
                            if chunk_overlap > 0 and len(current_lines) > 1:
                                current_tokens = 0
                                keep_idx = len(current_lines)
                                while keep_idx > 1 and current_tokens < chunk_overlap:
                                    keep_idx -= 1
                                    current_tokens += current_counts[keep_idx]
                                current_lines = current_lines[keep_idx:]
                                current_counts = current_counts[keep_idx:]
                                current_tokens = sum(current_counts)
                            else:
                                current_lines = []
                                current_counts = []
                                current_tokens = 0
                        # add line to current (original text)
                        current_lines.append(ln)
                        current_counts.append(t_count)
                        current_tokens += t_count

                # flush last chunk
                if current_lines:
                    chunks.append("\n".join(current_lines))
            else:
                raise ValueError(f"Unknown format: {format}. Will fallback to naive chunking.")
        except Exception as e:
            # If parsing fails for any reason, fallback to naive chunking
            print(f"Warning: Chunking with format {format} failed due to {e}. Falling back to naive chunking.")
            spans = LLMReviewer._tok_spans(text)
            total_tokens = len(spans)
            if total_tokens == 0:
                return ""
            step = chunk_token_num
            for i in range(0, total_tokens, step):
                j = min(i + chunk_token_num, total_tokens)
                start_char = spans[i][0]
                end_char = spans[j - 1][1]
                chunks.append(text[start_char:end_char])

        if not chunks:
            return ""

        # Optional safety: avoid extremely large batch to the embedding API
        if len(chunks) > max_chunks and prefilter_by_keywords and keywords:
            # Pre-filter: keep chunks containing any keyword first
            kw_chunks = [c for c in chunks if Reviewer.str_contains_keyword(c, keywords) != []]
            # If still too many, trim
            chunks = (kw_chunks or chunks)[:max_chunks]
        elif len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]

        # II) Keyword score per chunk
        # Simple score: number of distinct keywords present; then normalize by max across chunks
        kw_scores = np.array([len(Reviewer.str_contains_keyword(c, keywords)) for c in chunks])

        # III) Embedding similarity
        # Use OpenAI embeddings: embed the query built from keywords and each chunk
        try:
            if embedding_backend == "openai":
                # Create embeddings in two calls: one for query, one for chunks
                # q_emb_resp = openai.embeddings.create(model=emb_model, input=[query])
                q_emb_resp = self.client.embeddings.create(model=emb_model, input=keywords)
                query_emb = np.asarray([d.embedding for d in q_emb_resp.data], dtype=np.float32)
                ch_emb_resp = self.client.embeddings.create(model=emb_model, input=chunks)
                chunk_embs = np.asarray([d.embedding for d in ch_emb_resp.data], dtype=np.float32)
            else:
                model_name = LLMReviewer._rag_embedding_params[embedding_backend]["model_name"]
                if not model_name:
                    raise ValueError(f"No embedding model configured for backend '{embedding_backend}'.")
                embedder = LLMReviewer._get_embedder(model_name)
                encode_kwargs = {"convert_to_numpy": True}
                # query_emb = embedder.encode([query], **encode_kwargs).astype(np.float32)
                query_emb = embedder.encode(keywords, **encode_kwargs).astype(np.float32)
                chunk_embs = embedder.encode(chunks, **encode_kwargs).astype(np.float32)
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            chunk_embs = chunk_embs / np.linalg.norm(chunk_embs, axis=1, keepdims=True)
            sim_scores = chunk_embs @ query_emb.T
            sim_scores = np.max(sim_scores, axis=1)
        except Exception as e:
            # If embedding backend fails, fall back to keyword-only scoring
            print(f"Embedding backend '{embedding_backend}' failed: {e}. Falling back to keyword-only scores.")
            sim_scores = np.zeros(len(chunks), dtype=np.float32)
        sim_scores = (1 + sim_scores) / 2.0  # map cosine from [-1,1] to [0,1]

        sim_scores_high = sim_scores[kw_scores > 0]
        sim_scores_low = sim_scores[kw_scores == 0]
        self.rag_stats["keyword_mean"].append(np.mean(kw_scores))
        self.rag_stats["keyword_std"].append(np.std(kw_scores))
        self.rag_stats["embedding_mean"].append(np.mean(sim_scores))
        self.rag_stats["embedding_std"].append(np.std(sim_scores))
        self.rag_stats["embedding_high_mean"].append(np.mean(sim_scores_high) if len(sim_scores_high) > 0 else 0.0)
        self.rag_stats["embedding_high_std"].append(np.std(sim_scores_high) if len(sim_scores_high) > 0 else 0.0)
        self.rag_stats["embedding_low_mean"].append(np.mean(sim_scores_low) if len(sim_scores_low) > 0 else 0.0)
        self.rag_stats["embedding_low_std"].append(np.std(sim_scores_low) if len(sim_scores_low) > 0 else 0.0)
        self.rag_stats["embedding_separation"].append(normal_seperation(np.mean(sim_scores_high), np.std(sim_scores_high), np.mean(sim_scores_low), np.std(sim_scores_low)))
        if verbose:
            print("RAG Keyword + Embedding Retrieval Stats:")
            print(f"Keywords: {np.mean(kw_scores):.3f} ± {np.std(kw_scores):.3f}")
            print(f"Embeddings: {np.mean(sim_scores):.3f} ± {np.std(sim_scores):.3f}")
            print(f"Similarity score threshold: {(np.mean(sim_scores_high) + np.mean(sim_scores_low)) / 2:.3f}")
            print(f"Similarity scores for chunks WITH keywords: {np.mean(sim_scores_high):.3f} ± {np.std(sim_scores_high):.3f}")
            print(f"Similarity scores for chunks WITHOUT keywords: {np.mean(sim_scores_low):.3f} ± {np.std(sim_scores_low):.3f}")
            print("Relative deviation for two categories:", 2 * (np.mean(sim_scores_high) - np.mean(sim_scores_low)) / (np.std(sim_scores_high) + np.std(sim_scores_low)))

        # IV) Weighted sum and ranking
        # Normalize weights to sum to 1 for stable interpretation
        combined = keyword_weight * kw_scores + embedding_weight * sim_scores

        # Select chunks passing threshold; keep original order
        selected_idx = [i for i, s in enumerate(combined) if s >= similarity_threshold]
        if not selected_idx:
            # Fallback to top-k by score, but present in original document order for readability
            top_k = min(len(keywords), len(chunks))
            selected_idx = np.sort(np.argsort(-combined)[:top_k])
        rag_snippets = [chunks[i] for i in sorted(selected_idx)]

        return rag_snippets


    def search_text(self, model: str, exp_calc: Literal["exp", "calc"], prompt: str, schema: dict) -> str:
        """
        Search the given URL for relevant information using the OpenAI API.
        Returns the reply from the OpenAI API.

        The only entry for calling LLM API in this script.
        """
        if (len(prompt) > 120000):
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"Warning: The prompt has {len(prompt)} characters, which is too long. Skipping. Please manually check this paper.\n\n")
            print(f"Warning: The prompt has {len(prompt)} characters, which is too long. Skipping. Please manually check this paper.")
            return ''
        try:
            messages = [
                {
                    "role": "system", 
                    "content": getattr(LLMReviewer, f"static_prompt_{exp_calc}")
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ResponseSchema",
                    "strict": True,
                    "schema": schema
                }
            }
            completion = self.client.responses.create(
                model=model,
                input=messages,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "ResponseSchema",
                        "schema": schema,
                        "strict": True,
                    }
                },
                max_output_tokens=6000,
            )
            reply = completion.output_text.strip()

            # Token usage (may be None depending on API/model/config)
            usage = getattr(completion, "usage", None)
            tokens = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None
            }

            with open(self.llm_log_file, "a", encoding="utf-8") as f:
                f.write('--------------------------------\n')
                f.write(f"Model: {model}\n")
                f.write(f"Messages: {messages}\n")
                f.write(f"Reply: {reply}\n")
                f.write("--------------------------------\n")
            return reply, tokens
        except Exception as e:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"OpenAI API error: {e}\n")
            return '', None
    

    def search_pdf(self, model: str, exp_calc: Literal["exp", "calc"], prompt: str, pdf_path: str):
        """
        Search the given PDF for relevant information using the OpenAI API.
        Returns the reply from the OpenAI API.

        The OpenAI API put into the model's context both extracrted text and an image of each page, regardless of whether the page include images.
        Before deploying this, ensure you understand the pricing and token usage implications of using PDFs as input.

        No test is done at the moment, but ChatGPT-5 thinking estimates the cost to be 1.7x to 4x.
        """
        with open(pdf_path, "rb") as f:
            data = f.read()
        base64_encoded_pdf = base64.b64encode(data).decode("utf-8")
        try:
            messages = [
                {
                    "role": "system", 
                    "content": getattr(LLMReviewer, f"static_prompt_{exp_calc}")
                },
                {
                    "role": "user",
                    "content": 
                    [
                        {
                            "type": "text",
                            "content": prompt
                        },
                        {
                            "type": "file",
                            "file": {
                                "filename": pdf_path,
                                "file_data": f"data:application/pdf;base64,{base64_encoded_pdf}"
                            }
                        }
                    ]
                }
            ]
            completion = self.client.responses.create(
                model=model,
                input=messages,
                max_output_tokens=6000,
            )
            reply = completion.output_text.strip()
            with open(self.llm_log_file, "a", encoding="utf-8") as f:
                f.write('--------------------------------\n')
                f.write(f"Model: {model}\n")
                f.write(f"Messages: {messages}\n")
                f.write(f"Reply: {reply}\n")
                f.write("--------------------------------\n")
            return reply
        except Exception as e:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"OpenAI API error: {e}\n")
            return ''


    def search_one(self, paper: dict, using_url: bool = True, using_pdf: bool = True) -> float:
        """
        Review one paper.
        """
        raise NotImplementedError("search_one is not implemented")
        if (paper == {}):
            print(f"Warning: Paper is empty. Skipping...")
            return 0
        confidence_score = 0
        if using_url and paper["publication_link"]:
            response = self.search_text(self.model1, f"{LLMReviewer.first_prompt}\n\nCSD reference code of the crystal of interest: {self.id}\nPaper URL: {paper['publication_link']}")
            try:
                confidence_score_1 = float(response)
            except Exception:
                print(f"Could not parse confidence score from model 1 reply: {response}")
            else:
                if confidence_score_1 > self.threshold:
                    response = self.search_text(self.model2, f"{LLMReviewer.second_prompt}\n\nCSD reference code of the crystal of interest: {self.id}\nPaper URL: {paper['publication_link']}")
                    try:
                        response_json = json.loads(response)
                        confidence_score_2 = float(response_json["confidence_score"])
                        with open(self.output_file, "a", encoding="utf-8") as f:
                            f.write(f"Optical gap: {response_json['optical_gap']}\n")
                            f.write(f"Found in {response_json['content']} from {paper['publication_link']} \
                                with {self.model1} confidence score {confidence_score_1} and {self.model2} confidence score {confidence_score_2}\n")
                    except Exception as e:
                        print(f"Could not parse JSON from model 2 reply: {response}\nError: {e}")
                    confidence_score = max(confidence_score, confidence_score_2)
        if using_pdf and paper["pdf_link"]:
            raise NotImplementedError("search_pdf is not implemented")
        return confidence_score



if __name__ == "__main__":
    service = Service("C:/Users/18000/OneDrive/Desktop/VSCode/chromedriver-win64/chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    reviewer = KeyWordSearch(driver=driver)
    text = reviewer.pdf_link_to_text("https://www.osti.gov/pages/servlets/purl/2546930", "https://www.nature.com/articles/s43586-024-00293-8")
    driver.close()
