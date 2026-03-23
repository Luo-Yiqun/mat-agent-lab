"""
@ Yiqun Luo (luo2@andrew.cmu.edu)

This script defines a class to get the citing papers of a given CSD reference code.

This code only uses Google Scholar and does not need special access to publishers.

STRONGLY recommend to use selenium to avoid access forbidden (403) error, although it cannot be avoided sometimes.
If you want to use selenium, please download and install the chromedriver and set the path to the chromedriver.
You may need to manually answer robot verification, especially for logining to Google Scholar for the first time.
"""
# Todo:
# get PDF link, maybe Google search title



import time
import os
import shutil
import random
from tqdm import tqdm

import requests
from urllib.parse import urlparse, urlsplit, parse_qs, unquote, quote_plus, urljoin
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

try:
    from ccdc.io import EntryReader
    from ccdc.crystal import PackingSimilarity
except ImportError as e:
    print(e, "No CCDC Python API found.")

from utilities import *



class CCDCCitingPaper:
    """
    A class to get the citing papers of a given CSD reference code, including the title, abstract, website link, and pdf link, if available.
    """
    def generate_google_scholar_query(paper: dict[str, str]) -> str:
        """
        Generate a Google Scholar query string from the given paper information.
        """
        if paper.get("doi", ''):
            return paper["doi"]
        else:
            return f"{paper.get('authors', '')}, {paper.get('journal', '')}, {paper.get('year', '')}, {paper.get('volume', '')}, {paper.get('first_page', '')}"


    def __init__(self,
                 refcode: str, 
                 driver: webdriver.Chrome = None,
                 json_file: str = None,
                 include_other_ids: bool = True,
                 rmsd_threshold: float = 0.5,
                 ) -> None:
        self.refcode = refcode
        self.include_other_ids = include_other_ids
        self.rmsd_threshold = rmsd_threshold
        self.json_file = json_file
        if driver:
            self.driver = driver

        self.refcodes = [self.refcode]
        if self.include_other_ids:
            self.refcodes = self.get_all_ids()
        csd = EntryReader('CSD')
        self.entry = csd.entry(self.refcode)

        self.original_papers = []
        self.citing_papers = []
        self.n_citing_papers_google_scholar = None
        self.n_citing_papers_json = None

    def same_polymorph(self, other_refcode: str) -> bool:
        """
        Check if the given refcode and other_refcode are the same polymorphs based on the space group and RMSD.
        Returns True if they are the same polymorphs, False otherwise.
        """
        csd = EntryReader('CSD')
        try:
            e1 = csd.entry(self.refcode)             # your refcode
            e2 = csd.entry(other_refcode)            # other refcode
            if e1.crystal.spacegroup_number_and_setting == e2.crystal.spacegroup_number_and_setting:
                similarity_engine = PackingSimilarity()
                similarity_engine.settings.packing_shell_size = 20
                h = similarity_engine.compare(e1.crystal, e2.crystal)
                if h is None:
                    similarity_engine.settings.allow_molecular_differences = True
                    h = similarity_engine.compare(e1.crystal, e2.crystal)
                if h and h.rmsd is not None:
                    if h.rmsd < self.rmsd_threshold:
                        return True
        except Exception as e:
            print(f"Error: {e}")
                
        return False
    

    def get_all_ids_from_ccdc(self) -> list[str]:
        """
        Get all CCDC IDs associated with the given polymorph.
        Returns a list of CCDC IDs.
        """
        csd = EntryReader('CSD')
        
        similarity_engine = PackingSimilarity()
        similarity_engine.settings.packing_shell_size = 20

        prefix = self.refcode[:6]
        if prefix != self.refcode and self.same_polymorph(prefix):
            self.refcodes.append(prefix)

        postfix = 0
        while True:
            postfix += 1
            other_refcode = f"{prefix}{postfix:02d}"
            try:
                csd.entry(other_refcode)
            except RuntimeError as e:
                break
            if other_refcode not in self.refcodes and self.same_polymorph(other_refcode) :
                self.refcodes.append(other_refcode)

        print(f"{len(self.refcodes)} found for the original CSD reference code {self.refcode}: {self.refcodes}")
        return self.refcodes


    def write_all_ids_to_json(self, overwrite: bool = False) -> None:
        if self.json_file:
            if os.path.exists(self.json_file):
                with open(self.json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}
            if self.refcode in data:
                if "refcodes" in data[self.refcode]:
                    if overwrite:
                        print(f"Warning: {self.refcode}/refcodes already exists in {self.json_file}. Will overwrite it!")
                        data[self.refcode]["refcodes"] = self.refcodes
                    else:
                        print(f"Warning: {self.refcode}/refcodes already exists in {self.json_file}. Will not overwrite it!")
                else:
                    data[self.refcode]["refcodes"] = self.refcodes
            else:
                data[self.refcode] = {"refcodes": self.refcodes}
            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            print("Warning: self.json_file is None. Not able to write to it.")
        return


    def get_all_ids_from_json(self) -> bool:
        if len(self.refcodes) > 1:
            print("Warning: self.refcodes already contains multiple refcodes. Will overwrite it from JSON file!")
        if self.json_file and os.path.exists(self.json_file):
            with open(self.json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if self.refcode in data:
                if "refcodes" in data[self.refcode]:
                    refcodes = data[self.refcode]["refcodes"]
                    if refcodes[0] == self.refcode:
                        self.refcodes = refcodes
                        return True
                    else:
                        print(f"Warning: The first refcode in {self.refcode}/refcodes is not {self.refcode}. Will not rewrite self.refcodes and return false.")
                        return False
                else:
                    print(f"Warning: {self.refcode}/refcodes not found in {self.json_file}. Will return the original self.refcodes.")
                    return False
            else:
                print(f"Warning: {self.refcode} not found in {self.json_file}. Will return the original self.refcodes.")
                return False
        else:
            print(f"Warning: self.json_file {self.json_file} does not exist. Will return the original self.refcodes.")
            return False


    def get_all_ids(self) -> list[str]:
        if self.include_other_ids:
            if not self.get_all_ids_from_json():
                self.get_all_ids_from_ccdc()
                self.write_all_ids_to_json(True)
        return self.refcodes


    def get_original_papers_from_ccdc(self) -> list[str]:
        """
        Get the original papers of self.refcodes from the CSD database.

        For each original paper, it include authors, journal, volume, year, first_page, doi.
        """
        csd = EntryReader('CSD')
        if isinstance(self.original_papers, list) and len(self.original_papers) > 0:
            print("Warning: self.original_papers is not empty. Will overwrite it!")
        self.original_papers = []

        for refcode in self.refcodes:
            try:
                e = csd.entry(refcode)
            except Exception as e:
                print(f"Error: {e}")
            else:
                pubs = e.publications                 # publication
                for i in pubs:
                    paper_info = {
                        "refcode": refcode,
                        "authors": i.authors,
                        "journal": i.journal.full_name,
                        "volume": i.volume,
                        "year": str(i.year),
                        "first_page": i.first_page,
                        "doi": i.doi
                    }
                    if not any(all(paper_info[key] == existing_paper[key] for key in paper_info if key != "refcode") for existing_paper in self.original_papers):
                        self.original_papers.append(paper_info)
                    else:
                        print(f"Duplicate original paper found for {refcode}: {paper_info}. Will not add it again.")

        return self.original_papers

    
    def write_original_papers_to_json(self, overwrite: bool = False) -> None:
        """
        Write the original papers to a JSON file.
        """
        if self.original_papers == []:
            print(f"Warning: No original paper found for {self.refcode}. Will not write to {self.json_file}!")
            return
        if self.json_file:
            if os.path.exists(self.json_file):
                with open(self.json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}
            if self.refcode in data:
                if "original_papers" in data[self.refcode]:
                    if overwrite:
                        print(f"Warning: {self.refcode}/original_papers already exists in {self.json_file}. Will overwrite it, including citing papers!")
                        data[self.refcode]["original_papers"] = self.original_papers
                    else:
                        print(f"Warning: {self.refcode}/original_papers already exists in {self.json_file}. Will not overwrite it!")
                else:
                    data[self.refcode]["original_papers"] = self.original_papers
            else:
                data[self.refcode] = {"original_papers": self.original_papers}
            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            print("Warning: self.json_file is None. Not able to write to it.")
        return
    

    def get_original_papers_from_json(self) -> list[str]:
        """
        Get the original papers from the JSON file.
        Returns a list of original papers, which probably only contains 1.
        if not found, return None.
        """
        if self.json_file and os.path.exists(self.json_file):
            with open(self.json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if self.refcode in data:
                if "original_papers" in data[self.refcode]:
                    self.original_papers = data[self.refcode]["original_papers"]
                else:
                    print(f"Warning: {self.refcode}/original_papers not found in {self.json_file}. Will return None!")
                    self.original_papers = None
            else:
                print(f"Warning: {self.refcode} not found in {self.json_file}. Will return None!")
                self.original_papers = None
        else:
            print(f"Warning: self.json_file {self.json_file} does not exist. Will return None!")
            self.original_papers = None
        return self.original_papers
    

    def get_original_papers(self) -> list[str]:
        """
        Get the original papers.
        It will automatically get the original papers from the JSON file if available, otherwise from the CSD database.
        Returns a list of original papers, which probably only contains 1.
        """
        if self.json_file and os.path.exists(self.json_file):
            self.original_papers = self.get_original_papers_from_json()
            if self.original_papers:
                print(f"Original papers for {self.refcode} obtained from JSON file: {self.original_papers}")
                return self.original_papers
        self.original_papers = self.get_original_papers_from_ccdc()
        print(f"Original papers for {self.refcode} obtained from CSD database: {self.original_papers}")
        self.write_original_papers_to_json(True)
        return self.original_papers
    

    def get_n_citing_papers_from_google_scholar(self, restart_if_error: bool = True) -> int:
        """
        Get the number of citing papers from Google Scholar.

        restart_if_error: whether to restart the driver and fetch the page again if failed to load the Google Scholar page.
        This setting normally only makes a difference when one tries to fetch too many polymorphs in a single run.
        The missing rate may go up from ~1% for the first polymorph to ~9% for later ones.
        If one does not want to miss any paper, please set it to be False and restart a new process.
        If one wants the process to run without any interruption and does not mind to miss some paper, set it to be True.
        """
        if isinstance(self.n_citing_papers_google_scholar, int) and self.n_citing_papers_google_scholar > 0:
            print("Warning: self.n_citing_papers_google_scholar is not None. Will overwrite it!")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        self.n_citing_papers_google_scholar = len(self.original_papers)
        for paper in self.original_papers:
            search_url = f"https://scholar.google.com/scholar?q={CCDCCitingPaper.generate_google_scholar_query(paper)}"
            if self.driver:
                if restart_if_error:
                    try:
                        self.driver.get(search_url)
                        WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#gs_res_ccl")))
                    except Exception:
                        print(f"Error: Failed to load Google Scholar page for {self.refcode}. Will try to restart the driver and fetch the page again.")
                        try:
                            _service_path = getattr(getattr(self.driver, "service", None), "path", None)
                            _options = getattr(self.driver, "options", None)
                            self.driver.quit()
                            time.sleep(random.uniform(1, 3))
                        except Exception as e:
                            print(f"Error restarting the driver for {self.refcode}: {e}")
                            self.driver = None
                            r = requests.get(search_url, headers = headers)
                        self.driver = webdriver.Chrome(service=Service(_service_path), options=_options)
                else:
                    self.driver.get(search_url)
                    WebDriverWait(self.driver, 40).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#gs_res_ccl")))
                html = self.driver.page_source
            else:
                r = requests.get(search_url, headers = headers)
                html = r.text
            soup = BeautifulSoup(html, "html.parser")

            result = soup.find_all("div", class_ = lambda c: c and all(k in c for k in ["gs_r", "gs_or", "gs_scl"]))
            if len(result) == 0:
                print(f"Warning: No results found for paper {paper['refcode']} in Google Scholar. It might be some website fetch issue.")
                continue
            if len(result) > 1:
                print(f"Warning: Found multiple results for paper {paper['refcode']} in Google Scholar. Will only use the first one. Please MANUALLY CHECK the results if needed!")

            for a in result[0].find_all("a"):
                if a.text.startswith("Cited by"):
                    tmp = re.search("Cited by\s+(\d+)", a.text)
                    if tmp:
                        self.n_citing_papers_google_scholar += int(tmp.group(1))
                        break
        return self.n_citing_papers_google_scholar
    

    def get_n_citing_papers_from_json(self) -> int:
        """
        Get the number of citing papers from a JSON file.
        """
        if isinstance(self.n_citing_papers_json, int) and self.n_citing_papers_json > 0:
            print("Warning: self.n_citing_papers_json is not None. Will overwrite it!")

        if self.json_file and os.path.exists(self.json_file):
            with open(self.json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if self.refcode in data and "citing_papers" in data[self.refcode]:
                self.n_citing_papers_json = len(data[self.refcode]["citing_papers"])
            else:
                print(f"Warning: {self.refcode}/citing_papers not found in {self.json_file}. Will return 0.")
                self.n_citing_papers_json = 0
        else:
            print(f"Warning: self.json_file {self.json_file} does not exist. Will return 0.")
            self.n_citing_papers_json = 0
        return self.n_citing_papers_json


    def google_scholar_single_result(self, res) -> dict:
        """
        Extract the title, link, snippet, and PDF link from a single Google Scholar result.
        Returns a dict with the title, link, snippet, and PDF link.
        """
        title_tag = res.find("h3", class_="gs_rt")
        title = title_tag.text if title_tag else ""
        link = title_tag.a["href"] if title_tag and title_tag.a else ""
        # Sometimes the snippet contains the abstract, but Google Scholar does not provide a true abstract.
        # We'll use the snippet as a proxy for the abstract if available.
        snippet_tag = res.find("div", class_="gs_rs")
        snippet = snippet_tag.text if snippet_tag else ""
        pdf_tag = res.find("div", class_="gs_or_ggsm")
        pdf_link = pdf_tag.a["href"] if pdf_tag and pdf_tag.a else ""
        # Add these to the output dict
        original_paper = {
            "publication_link": link,
            "pdf_link": pdf_link,
            "google_scholar_title": title,
            "google_scholar_snippet": snippet,
        }
        return original_paper
    

    def arxiv_single_title_search(self, title: str) -> dict:
        """
        Search for a paper on Arxiv by title.
        Returns a dict with the title, link, snippet, and PDF link.
        """
        raise NotImplementedError("Not implemented yet")
        search_url = f"https://arxiv.org/search/?query={urllib.parse.quote(title)}"
        r = requests.get(search_url)
        soup = BeautifulSoup(r.text, "html.parser")
        result = soup.find_all("div", class_="gs_ri")
        if len(result) > 1:
            print(f"Warning: Found multiple results for {title} in Arxiv. Will only use the first one. Please MANUALLY CHECK the results if needed!")
        item = {}
        # Find the first result (if any)


    def chemrxiv_single_title_search(self, title: str) -> dict:
        """
        Search for a paper on ChemRxiv by title.
        Returns a dict with the title, link, snippet, and PDF link.
        """
        raise NotImplementedError("Not implemented yet")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        search_url = f"https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/search?query={urllib.parse.quote(title)}"
        r = requests.get(search_url, headers=headers, timeout=10)
        if r.status_code != 200:
            print(f"ChemRxiv search failed with status {r.status_code} for title: {title}")
            return {}
        soup = BeautifulSoup(r.text, "html.parser")
        result = soup.find_all("div", class_="MatchResult v-div mb-4 border-bottom border-normal border-softened")
        if len(result) > 1:
            print(f"Warning: Found multiple results for {title} in ChemRxiv. Will only use the first one. Please MANUALLY CHECK the results if needed!")
        item = {}
        # Find the first result (if any)
        if result:
            first = result[0]
            # Title
            title_tag = first.find("div", class_="match-overview__title-container")
            item["chemrxiv_title"] = title_tag.select_one("h3 span").get_text(strip=True) if title_tag else None
            if not self.close_to_title(title, item["chemrxiv_title"]):
                return {}
            # Link
            item["chemrxiv_link"] = "https://chemrxiv.org" + first.select_one("article a")["href"]
            if item["chemrxiv_link"]:
                chemrxiv_response = requests.get(item["chemrxiv_link"], headers=headers, timeout=10)
                if chemrxiv_response.status_code != 200:
                    print(f"Failed to fetch ChemRxiv paper page with status {chemrxiv_response.status_code}")
                    return item
                soup = BeautifulSoup(chemrxiv_response.text, "html.parser")
                abstract_tag = soup.find("div", class_="abstract")
                item["chemrxiv_abstract"] = abstract_tag.get_text(strip=True) if abstract_tag else None
                doi_tag = soup.find('a', href = lambda x: x and "doi.org" in x)
                item["chemrxiv_doi"] = doi_tag.get_text(strip=True) if doi_tag else None
        return item


    def google_scholar_citer_from_paper(self, paper: dict[str, str]) -> list[dict]:
        """
        Get the citing papers from Google Scholar by URLs to the original experimental paper.
        Returns a list of dicts with basic info (link, title, abstract, snippet).
        Note: Google Scholar does not provide an official API and may block automated requests.
        """
        citing_papers = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        url = f"https://scholar.google.com/scholar?q={CCDCCitingPaper.generate_google_scholar_query(paper)}"
        if self.driver:
            self.driver.get(url)
            try:
                WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#gs_res_ccl")))
            except Exception:
                pass
            html = self.driver.page_source
        else:
            r = requests.get(url, headers=headers)
            html = r.text
        soup = BeautifulSoup(html, "html.parser")
        
        result = soup.find_all("div", class_ = lambda c: c and all(k in c for k in ["gs_r", "gs_or", "gs_scl"]))
        if len(result) == 0:
            print(f"Warning: No results found for {url} in Google Scholar. It might be some website fetch issue.")
            return []
        if len(result) > 1:
            print(f"Warning: Found multiple results for {url} in Google Scholar. Will only use the first one. Please MANUALLY CHECK the results if needed!")
        self.citing_papers.append(self.google_scholar_single_result(result[0]))

        cited_by_url = None
        for a in soup.find_all("a"):
            if a.text.startswith("Cited by"):
                cited_by_url = "https://scholar.google.com" + a.get("href")
                break
        if not cited_by_url:
            return self.citing_papers

        start = 0
        while True:
            paged_url = cited_by_url + f"&start={start}"
            if self.driver:
                self.driver.get(paged_url)
                try:
                    WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#gs_res_ccl")))
                except Exception:
                    pass
                html = self.driver.page_source
            else:
                r = requests.get(paged_url, headers=headers)
                html = r.text
            soup = BeautifulSoup(html, "html.parser")
            results = soup.find_all("div", class_ = lambda c: c and all(k in c for k in ["gs_r", "gs_or", "gs_scl"]))
            if not results:
                break
            for res in results:
                citing_papers.append(self.google_scholar_single_result(res))
            start += 10
            time.sleep(2.0)
        
        return citing_papers
    

    def google_scholar_citer(self) -> list[dict]:
        """
        Get the citing papers from Google Scholar by DOI.
        Returns a list of dicts with basic info (link, title, abstract, snippet).
        """
        if self.original_papers == []:
            print(f"Warning: No original papers found for {self.refcode}")
        if isinstance(self.citing_papers, list) and len(self.citing_papers) > 0:
            print("Warning: self.citing_papers is not empty. Will overwrite it!")
        self.citing_papers = []

        for paper in self.original_papers:
            self.citing_papers.extend(self.google_scholar_citer_from_paper(paper))

        if len(self.citing_papers) > 500:
            print("More than 500 citing papers found for {self.refcode} from Google!")
        return self.citing_papers


    def openalex_citers(self):
        """
        Get the citing papers from OpenAlex.
        Returns a list of dicts with basic info (title, link, snippet).
        
        Note: OpenAlex is more friendly for scraping.
        Not recommended: OpenAlex seems to provide fewer citing papers than Google Scholar.
        """
        raise NotImplementedError("Not implemented yet")
    

    def get_pdf_link(self, paper: dict) -> dict:
        """
        Get the PDF link of a paper from its title by Google search (suggested by Noa) or some other ways.
        
        Input: paper, a dict with the following keys
        {
            "publication_link": <publication_link>,
            "pdf_link": '',
            "google_scholar_title": <title>,
            "google_scholar_snippet": <snippet>,
        }
        where <publication_link>, <title>, <snippet> are almost always available, and <snippet> is normally part of the abstract
        Output: still this paper dict, but with the "*pdf_link" updated if found.

        In the normal pipeline, this is not used in CCDCCitingPaper.google_scholar_citer_from_doi or other Google scholar functions to avoid interrupting the normal workflow.
        This is normally used in CCDCCitingPapers.json_citer_read.

        You may use some open-source paper websites, not just Google search.
        You may also use <publication_link> and/or <snippet> in addition to <title>.
        """
        PDF_CT = "application/pdf"
        UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")

        # ---------- utils ----------
        def _clean_url(u: str | None) -> str | None:
            if not u: return u
            return u.split("#")[0].strip()

        def _looks_like_pdf_bytes(chunk: bytes) -> bool:
            # return true if the bytes look like a PDF file (by magic number)
            return bool(chunk and chunk.lstrip().startswith(b"%PDF"))

        def _open_pdf_direct(url: str, referer: str | None = None, timeout=10) -> str | None:
            """Check if the URL directly points to a PDF by inspecting headers and content.
            """
            if not url:
                return None
            url = _clean_url(url)
            try:
                headers = {"User-Agent": UA, "Accept": PDF_CT}
                if referer: headers["Referer"] = referer
                r = requests.get(url, headers=headers, stream=True,
                                allow_redirects=True, timeout=timeout)
                # read the headers
                ct = (r.headers.get("Content-Type") or "").lower()
                # take a peek at the content
                buf = b""
                it = r.iter_content(8192)
                for _ in range(4):
                    try:
                        chunk = next(it)
                    except Exception:
                        break
                    if not chunk: break
                    buf += chunk
                    if len(buf) >= 32768: break

                # check if the content looks like a PDF
                if _looks_like_pdf_bytes(buf):
                    return r.url
                if ("pdf" in ct) and (b"<html" not in buf.lower()):
                    return r.url
            except requests.RequestException:
                pass
            return None

        def _fetch_html(url: str, timeout=12) -> str | None:
            try:
                if self.driver:
                    self.driver.get(url)
                    return self.driver.page_source
                else:
                    r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
                    if r.status_code == 200 and "text/html" in (r.headers.get("Content-Type") or ""):
                        return r.text
            except Exception:
                pass
            return None

        def _extract_meta_pdf(html: str | None, base_url: str | None) -> str | None:
            """search pdf link in meta tags and <a> tags"""
            if not html: return None
            soup = BeautifulSoup(html, "html.parser")

            # 1) <meta name="citation_pdf_url">
            tag = soup.find("meta", attrs={"name": "citation_pdf_url"})
            if tag and tag.get("content"):
                href = tag["content"]
                if href.startswith("/") and base_url:
                    href = urljoin(base_url, href)
                real = _open_pdf_direct(href, referer=base_url)
                if real: return real

            # 2) <link rel="alternate" type="application/pdf">
            for link in soup.find_all("link", attrs={"rel": True, "type": True, "href": True}):
                rel = " ".join(link.get("rel", [])).lower()
                typ = (link.get("type") or "").lower()
                href = link.get("href")
                if "alternate" in rel and typ == PDF_CT and href:
                    if href.startswith("/") and base_url:
                        href = urljoin(base_url, href)
                    real = _open_pdf_direct(href, referer=base_url)
                    if real: return real

            # 3) <a href="*.pdf"> or text contains "pdf"
            for a in soup.find_all("a", href=True):
                txt = (a.get_text() or "").strip().lower()
                href = a["href"]
                if href.lower().endswith(".pdf") or "pdf" in txt:
                    if href.startswith("/") and base_url:
                        href = urljoin(base_url, href)
                    real = _open_pdf_direct(href, referer=base_url)
                    if real: return real
            return None

        DOI_RX = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.I)

        def _extract_doi(html: str | None, url: str | None) -> str | None:
            # extract DOI from html or url
            for src in (url or "", html or ""):
                m = DOI_RX.search(src)
                if m:
                    return m.group(0).rstrip(").,;")
            u = url or ""
            if "pubs.rsc.org" in (urlparse(u).netloc or ""):
                m = re.search(r"/content/articlehtml/\d+/\w+/([a-z0-9]+)", u, re.I)
                if m:
                    return f"10.1039/{m.group(1)}"
            return None

        def _doi_content_negotiation(doi: str, timeout=10) -> str | None:
            try:
                if self.driver:
                    self.driver.get(f"https://doi.org/{doi}")
                    final_url = self.driver.current_url
                    return _open_pdf_direct(final_url, timeout=timeout)
                else:
                    r = requests.get(f"https://doi.org/{doi}",
                                headers={"Accept": PDF_CT, "User-Agent": UA},
                                allow_redirects=True, timeout=timeout)
                    return _open_pdf_direct(r.url, timeout=timeout)
            except Exception:
                return None

        def _unpaywall(doi: str, email: str | None, timeout=10) -> str | None:
            if not email: return None
            api = f"https://api.unpaywall.org/v2/{doi}?email={email}"
            try:
                r = requests.get(api, headers={"User-Agent": UA}, timeout=timeout)
                if r.status_code == 200:
                    js = r.json()
                    locs = []
                    if js.get("best_oa_location"): locs.append(js["best_oa_location"])
                    locs.extend(js.get("oa_locations") or [])
                    for loc in locs:
                        for k in ("url_for_pdf", "url"):
                            u = _clean_url(loc.get(k))
                            if not u: continue
                            real = _open_pdf_direct(u, timeout=timeout)
                            if real: return real
            except requests.RequestException:
                pass
            return None

        def _repo_rule(pub: str | None, html: str | None) -> str | None:
            """rules for specific publishers / repositories"""
            if not pub: return None
            url = _clean_url(pub) or ""
            host = (urlparse(url).netloc or "").lower()
            path = (urlparse(url).path or "").lower()

            # RSC: /articlehtml/ -> /articlepdf/
            if "pubs.rsc.org" in host and "/content/articlehtml/" in path:
                cand = url.replace("/content/articlehtml/", "/content/articlepdf/")
                real = _open_pdf_direct(cand, referer=pub)
                if real: return real

            # ACS: /doi/abs|full/ -> /doi/pdf/
            if "pubs.acs.org" in host and "/doi/" in path:
                for cand in (
                    url.replace("/doi/abs/","/doi/pdf/").replace("/doi/full/","/doi/pdf/")+"?download=1",
                    url.replace("/doi/abs/","/doi/pdf/").replace("/doi/full/","/doi/pdf/"),
                ):
                    real = _open_pdf_direct(cand, referer=pub)
                    if real: return real

            # Wiley
            if "onlinelibrary.wiley.com" in host and "/doi/" in path:
                cand = url.replace("/doi/abs/","/doi/pdfdirect/").replace("/doi/full/","/doi/pdfdirect/")
                if not cand.endswith("?download=true"): cand += "?download=true"
                real = _open_pdf_direct(cand, referer=pub)
                if real: return real
                cand2 = url.replace("/doi/abs/","/doi/pdf/").replace("/doi/full/","/doi/pdf/")
                real = _open_pdf_direct(cand2, referer=pub)
                if real: return real

            # arXiv
            if "arxiv.org" in host:
                m = re.search(r"arxiv\.org/(abs|pdf)/([^/?#]+)", url, re.I)
                if m:
                    cand = f"https://arxiv.org/pdf/{m.group(2)}.pdf"
                    real = _open_pdf_direct(cand, referer=pub)
                    if real: return real

            return None

        # ------- search online with checks with title -------
        def _norm_title(t: str) -> str:
            t = t.lower()
            t = re.sub(r"[^a-z0-9\s\-\+\.\(\)]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        def _sim(a: str, b: str) -> float:
            try:
                import Levenshtein
                return Levenshtein.ratio(a, b)
            except Exception:
                from difflib import SequenceMatcher
                return SequenceMatcher(None, a, b).ratio()

        def _page_title(u: str) -> str | None:
            if u.lower().endswith(".pdf"):
                # use file name as title instead
                name = (urlparse(u).path or "").split("/")[-1]
                return re.sub(r"\.pdf$", "", name, flags=re.I)
            html = _fetch_html(u)
            if not html: return None
            soup = BeautifulSoup(html, "html.parser")
            meta = soup.find("meta", attrs={"name": "citation_title"})
            if meta and meta.get("content"): return meta["content"]
            if soup.title and soup.title.string: return soup.title.string
            return None

        def _strip_tracking(u: str) -> str:
            if not u: return u
            u = _clean_url(u)
            if not u: return u
            netloc, path = (urlparse(u).netloc or ""), (urlparse(u).path or "")
            if "google." in netloc and "/url" in path:
                qs = parse_qs(urlsplit(u).query)
                real = qs.get("q", [""])[0] or qs.get("url", [""])[0]
                return _clean_url(unquote(real)) or u
            if "bing.com" in netloc:
                qs = parse_qs(urlsplit(u).query)
                real = qs.get("u", [""])[0]
                try:
                    dec = unquote(real)
                    if dec.startswith("http"):
                        return _clean_url(dec)
                except Exception:
                    pass
            return u

        def _extract_links_from_search_html(html: str, engine: str) -> list[str]:
            if not html: return []
            soup = BeautifulSoup(html, "html.parser")
            links = []
            if engine == "ddg":
                for a in soup.select("a.result__a, a.result__url"):
                    if a.get("href"): links.append(a["href"])
            elif engine == "bing":
                for a in soup.select("li.b_algo h2 a, h2 a"):
                    if a.get("href"): links.append(a["href"])
            elif engine == "google":
                for a in soup.select("#search a"):
                    href = a.get("href")
                    if href and "accounts.google." not in href and not href.startswith("/search"):
                        links.append(href)
            out, seen = [], set()
            for u in links:
                u2 = _strip_tracking(u)
                if u2 and u2 not in seen:
                    seen.add(u2)
                    out.append(u2)
            return out

        def _engine_search(engine: str, query: str, timeout=10) -> list[str]:
            headers = {"User-Agent": UA}
            try:
                if engine == "ddg":
                    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
                elif engine == "bing":
                    url = f"https://www.bing.com/search?q={quote_plus(query)}&count=20&setlang=en"
                elif engine == "google":
                    url = f"https://www.google.com/search?q={quote_plus(query)}&num=10&hl=en"
                else:
                    return []
                if self.driver:
                    self.driver.get(url)
                    html = self.driver.page_source
                    return _extract_links_from_search_html(html, engine)
                else:
                    r = requests.get(url, headers=headers, timeout=timeout)
                    if r.status_code != 200: return []
                    return _extract_links_from_search_html(r.text, engine)
            except Exception:
                return []

        def _web_search_pdf_strict(title: str, pub_link: str | None, doi: str | None, max_candidates=25) -> str | None:
            queries = [f'"{title}" filetype:pdf', f'{title} filetype:pdf', f'"{title}" pdf']
            if doi:
                queries += [f'"{doi}" filetype:pdf', f'{doi} filetype:pdf', f'"{doi}" pdf']
            host = (urlparse(pub_link or "").netloc or "").lower()
            if title and host:
                queries += [f'site:{host} "{title}" filetype:pdf', f'site:{host} {title} filetype:pdf']

            engines = ["ddg","bing","google"]
            seen, cands = set(), []
            for q in queries:
                for eng in engines:
                    for u in _engine_search(eng, q):
                        u = _clean_url(u)
                        if not u: continue
                        if "sci-hub" in (u.lower()): continue
                        if u in seen: continue
                        seen.add(u); cands.append(u)
                        if len(cands) >= max_candidates: break
                    if len(cands) >= max_candidates: break
                if len(cands) >= max_candidates: break

            want = _norm_title(title)
            for u in cands:
                pt = _page_title(u) or ""
                if not pt: continue
                if _sim(_norm_title(pt), want) < 0.94:
                    continue
                real = _open_pdf_direct(u, referer=pub_link)
                if real: return real
            return None

        # ---------- pipeline ----------
        pub_link = _clean_url(paper.get("publication_link", ""))
        s_pdf    = _clean_url(paper.get("pdf_link", ""))
        title    = (paper.get("google_scholar_title") or "").strip()

        # 1) exist Google Scholar PDF linke
        if s_pdf:
            return paper

        # 2) publisher
        if pub_link:
            real = _open_pdf_direct(pub_link)
            if real:
                paper["pdf_link"] = real
                return paper

            html = _fetch_html(pub_link)
            real = _repo_rule(pub_link, html)
            if real:
                paper["pdf_link"] = real
                return paper

            real = _extract_meta_pdf(html, pub_link)
            if real:
                paper["pdf_link"] = real
                return paper
        else:
            html = None
        """
        # 3) DOI / unpaywall
        doi = _extract_doi(html, pub_link)
        if doi:
            real = _doi_content_negotiation(doi)
            if real:
                paper["pdf_link"] = real
                return paper
            email = getattr(self, "unpaywall_email", None) or os.getenv("UNPAYWALL_EMAIL")
            if email:
                real = _unpaywall(doi, email)
                if real:
                    paper["pdf_link"] = real
                    return paper

        # 4) title-based web search
        if title:
            real = _web_search_pdf_strict(title, pub_link, doi)
            if real:

                paper["pdf_link"] = real
                return paper
        """
        # 5) not found
        paper.setdefault("pdf_link","")

        return paper
        

    def update_pdf_link(self) -> None:
        """
        Update the PDF links of the citing papers if needed.
        """
        with open(self.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if self.refcode in data and "citing_papers" in data[self.refcode]:
                data = data[self.refcode]["citing_papers"]
            else:
                data = []
        title_to_pdf_link = {'': ''}
        for paper in data:
            title = paper.get("google_scholar_title", '')
            pdf_link = paper.get("pdf_link", '')
            if title and pdf_link:
                title_to_pdf_link[title] = pdf_link

        for paper in self.citing_papers:
            if paper.get("pdf_link", '') == '':
                if title_to_pdf_link.get(paper.get("google_scholar_title", ''), ''):
                    paper["pdf_link"] = title_to_pdf_link[paper.get("google_scholar_title", '')]
                else:
                    self.get_pdf_link(paper)
        return


    def json_citer_write(self) -> None:
        """
        Write the citing papers to a JSON file.
        """
        if self.json_file:
            if os.path.exists(self.json_file):
                with open(self.json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}
            if self.refcode in data:
                data[self.refcode]["citing_papers"] = self.citing_papers
            else:
                data[self.refcode] = {
                    "original_papers": self.original_papers,
                    "citing_papers": self.citing_papers
                    }
            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            print(f"Warning: self.json_file {self.json_file} does not exist. Will not write to it.")
        return
    

    def json_citer_read(self) -> list[dict]:
        """
        Read the citing papers from a JSON file.
        If not found, return None.
        """
        if self.json_file and os.path.exists(self.json_file):
            with open(self.json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if self.refcode in data and "citing_papers" in data[self.refcode]:
                self.citing_papers = data[self.refcode]["citing_papers"]
            return self.citing_papers
        else:
            print(f"Warning: self.json_file {self.json_file} does not exist. Will return None.")
            self.citing_papers = None
        return self.citing_papers


    def citer(self, restart_if_error: bool = True) -> list[dict]:
        """
        Get the citing papers from:
        I) self.json_file if valid
        II) Google Scholar
        """
        if self.get_n_citing_papers_from_google_scholar(restart_if_error) <= self.get_n_citing_papers_from_json():
            self.json_citer_read()
        else:
            print(f"More papers from Google Scholar found. Will update {self.json_file}")
            self.google_scholar_citer()
            self.update_pdf_link()
            self.json_citer_write()
        return self.citing_papers



if __name__ == "__main__":
    """
    paper = {
                "publication_link": "https://www.sciencedirect.com/science/article/pii/S0960894X11013138",
                "google_scholar_title": "A modular approach to trim cellular targets in anticancer drug discovery",
                "google_scholar_snippet": "A Phenotypic Drug Discovery strategy was applied to study a set of pyrimidine analogs prepared by means of intramolecular oxidation–reduction reactions of N-substituted-N-(2, 6 …",
                "pdf_link": ""
        }
    paper = {
                "publication_link": "https://pubs.acs.org/doi/abs/10.1021/acs.orglett.5c03674",
                "google_scholar_title": "Planar versus Twist: Two Conformers of a 5, 7, 12, 14-Tetrakis (triisopropylsilylethynyl) pentacene in the Solid State",
                "google_scholar_snippet": "We synthesized 4TIPS-pen, a pentacene derivative bearing TIPS-ethynyl groups at the 5, 7, 12, 14-positions via a modified route. The compound showed polymorphism, forming green …",
                "pdf_link": ""
        }
    service = Service("C:/Users/18000/OneDrive/Desktop/VSCode/chromedriver-win64/chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    c = CCDCCitingPaper(refcode = "AHEYUI", driver = driver)
    c.get_original_papers()
    c.get_pdf_link(paper)
    """
    """
    ids = [id.split('.')[0] for id in os.listdir("./json/sub1ev/")]
    service = Service("C:/Users/18000/OneDrive/Desktop/VSCode/chromedriver-win64/chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    for id in tqdm(ids, smoothing = 0.1):
        c = CCDCCitingPaper(refcode = id, driver = driver, json_file = "LiteratureReview/citing.json")
        c.get_original_papers()
        c.citer()
        shutil.copy("LiteratureReview/citing.json", "LiteratureReview/citing_back.json")
    driver.close()
    with open("LiteratureReview/citing.json", 'r', encoding = "utf-8") as f, open("LiteratureReview/citing_back.json", 'r', encoding = "utf-8") as f_backup:
        n_f = len(f.readlines())
        n_f_backup = len(f_backup.readlines())
    if n_f < n_f_backup:
        print("Restoring from backup...")
        shutil.copy("LiteratureReview/citing_back.json", "LiteratureReview/citing.json")
    """