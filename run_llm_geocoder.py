# run_llm_geocoder.py

import asyncio
import argparse
import getpass
import logging
import os
import sys
import time
from typing import List, Dict, Any

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

from config import PROCESSED_PMIDS_FILE
from cborg_loader import init_cborg_chat_model
from pubmed_loader import RobustPubMedLoader
from graph import build_llm_graph
from nodes import parser as output_parser  # renamed to avoid collision
from utils import explode_locations, load_processed_data, save_processed_data, dedupe_and_limit_rows

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler()],
)

# ----------------------------------------------------------------------
# Toggle geocoding on/off
# ----------------------------------------------------------------------
GEOCODE = False

if GEOCODE:
    from geocode_utils import add_geocode_columns as enrich_geocode
else:
    from geocode_utils import noop_geocode as enrich_geocode


# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------
PROMPT = """You are a biomedical data extraction assistant. The FULL TEXT of a scientific document is provided below. Extract information ONLY from this text — do not refuse or claim you cannot access it.

**TASK:** Extract structured information about *Burkholderia pseudomallei* / melioidosis from the document below.

---

## IMPORTANT: Eligibility Criteria

**INCLUDE only these study types:**
- **Natural human infections** – clinical case reports, outbreak investigations, hospital cohorts
- **Natural animal infections** – melioidosis in livestock, wildlife, or pets occurring naturally (not experimentally induced)
- **Environmental detections** – *B. pseudomallei* isolated from soil, water, or environmental samples in field studies

**EXCLUDE (mark as study_type = "Excluded"):**
- Laboratory experiments using animal models (e.g., "BALB/c mice", "mouse model", "experimental infection")
- In vitro studies (cell cultures, petri dish experiments)
- Vaccine development or drug efficacy trials in lab animals
- Genomic/bioinformatic analyses without field sampling
- Review articles, meta-analyses, or commentaries without original case data

**How to identify lab studies:**
- Mentions of lab animal strains: BALB/c, C57BL/6, CD-1 mice, Sprague-Dawley rats
- Phrases like: "experimentally infected", "challenge study", "animal model", "inoculated with"
- Institutional lab locations without field cases (e.g., "Porton Down", "BSL-3 facility")

---

## Fields to Extract

### 1. study_type
Classify into **exactly one** category (use exact string):
- `"Human cases"` – natural human melioidosis case(s) or outbreak
- `"Animal cases"` – natural melioidosis in animals (NOT lab experiments)
- `"Environmental cases"` – *B. pseudomallei* detected in environmental field samples
- `"Excluded"` – lab study, experimental infection, review, or otherwise ineligible
- `"unknown"` – cannot determine from text

### 2. sample_date
*Skip if study_type is "Excluded".*

Four-digit year of case occurrence or sample collection:
- Use explicitly stated year (e.g., "in 2017", "collected in 2020")
- If not stated, use publication year
- If undeterminable, use `"unknown"`

### 3. location
*Return empty array `[]` if study_type is "Excluded".*

Array of location objects for each natural case/sample site.

**Include only locations of natural occurrences:**
- Human cases → hospital, clinic, city where patient was treated (natural infection)
- Animal cases → farm, zoo, wildlife area where natural infection occurred
- Environmental cases → field sampling site (river, soil plot, rice paddy)

**Exclude:**
- Laboratory locations (research institutes, BSL facilities, animal facilities)
- Author affiliations
- Manufacturer sites

**Each location object must have these fields** (use `"unknown"` if not found):

| Field | Description |
|-------|-------------|
| `region` | Geographic region (e.g., "East Asia & Pacific", "South Asia") |
| `country` | Full country name for OpenStreetMap (e.g., "Thailand", "Australia") |
| `location` | Most specific place: city, district, hospital, or GPS coordinates |
| `amenity` | Type of place: "Hospital", "Farm", "Water source", etc. |
| `street` | Street address or coordinates as "lat, lon" |
| `city` | City or town name |
| `county` | County, district, or equivalent |
| `state` | State, province, or region |
| `postalcode` | Postal/ZIP code |

---

## Output Format

Return **only** valid JSON matching this schema — no explanations, no markdown fencing:

{format_instructions}

---

## DOCUMENT TEXT

{full_document}
"""

prompt_template = PromptTemplate(
    template=PROMPT,
    input_variables=["full_document"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
BATCH_SIZE = 5
MAX_DOCS = 10


def make_loader(cumulative_target: int) -> RobustPubMedLoader:
    return RobustPubMedLoader(
        query="melioidosis OR pseudomallei",
        max_docs=cumulative_target,
        fetch_full_text=True,
        processed_pmids_file=str(PROCESSED_PMIDS_FILE),
    )

async def process_one_doc(doc, graph) -> List[Dict[str, Any]]:
    """Returns a list of result rows – one per location occurrence."""
    pmid = doc.metadata["pmid"]
    logging.info(f"Processing PMID {pmid}")

    state = {
        "pmid": pmid,
        "full_document": doc.page_content,
        "doc_metadata": doc.metadata,
    }

    out = await graph.ainvoke(state)
    answer = out.get("answer", {})
    # Skip excluded studies
    if answer.get("study_type") == "Excluded":
        logging.info(f"PMID {pmid}: Excluded (lab study or ineligible)")
        return []  # No rows for this document
    
    accession_list = out.get("accession_numbers", [])

    base_result = {
        "pmid": pmid,
        "title": doc.metadata.get("title"),
        "study_type": answer.get("study_type"),
        "sample_date": answer.get("sample_date"),
        "retrieved_preview": doc.page_content[:500],
        "source": doc.metadata.get("source"),
        "location": answer.get("location", []),
        "accession_numbers": accession_list,
    }

    return explode_locations(base_result)

async def main(llm) -> None:
    """Main async pipeline."""
    graph = build_llm_graph(llm, prompt_template)
    
    processed_data = load_processed_data()
    total_processed = len(processed_data["pmids"])
    sem = asyncio.Semaphore(BATCH_SIZE)

    async def _process_one_doc_sema(doc):
        async with sem:
            try:
                return await process_one_doc(doc, graph)
            except Exception as exc:
                logging.error(f"Failed on PMID {doc.metadata['pmid']}: {exc}")
                return []

    batch_index = 0
    while total_processed < MAX_DOCS:
        remaining = MAX_DOCS - total_processed
        request_size = min(BATCH_SIZE, remaining)
        cumulative_target = total_processed + request_size

        loader = make_loader(cumulative_target)
        batch_docs = loader.load() or []
        if not batch_docs:
            logging.info("No more PubMed records to fetch – exiting.")
            break

        new_docs = [
            d for d in batch_docs if d.metadata["pmid"] not in processed_data["pmids"]
        ]
        if not new_docs:
            continue

        batch_index += 1
        logging.info(
            f"=== Processing batch {batch_index} ({len(new_docs)} new docs, "
            f"total processed so far: {total_processed})"
        )

        tasks = [_process_one_doc_sema(doc) for doc in new_docs]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        rows: List[Dict[str, Any]] = []
        for res in batch_results:
            if isinstance(res, Exception):
                continue
            rows.extend(res)

        rows = await enrich_geocode(rows)
        rows = dedupe_and_limit_rows(rows)

        if not rows:
            logging.info(f"Batch {batch_index} produced no rows – continuing.")
            continue

        #processed_data["pmids"].update(r["pmid"] for r in rows if "pmid" in r)
        #processed_data["results"].extend(rows)
        #save_processed_data(processed_data)

        for doc in new_docs:
            processed_data["pmids"].add(doc.metadata["pmid"])  # Mark ALL as processed

        # Then only add rows that have data:
        if rows:
            processed_data["results"].extend(rows)

        save_processed_data(processed_data)

        total_processed = len(processed_data["pmids"])
        logging.info(
            f"Batch {batch_index} finished – added {len(rows)} rows "
            f"(total rows stored: {len(processed_data['results'])})"
        )

        if total_processed >= MAX_DOCS:
            logging.info(f"Reached target of {MAX_DOCS} – stopping.")
            break

    logging.info("=== ALL DONE ===")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(  # renamed from 'parser'
        description="Run the llm_geocoder pipeline."
    )
    arg_parser.add_argument("--model", type=str, default=os.getenv("CBORG_MODEL", "openai/gpt-4o"))
    arg_parser.add_argument("--temperature", type=float, default=float(os.getenv("CBORG_TEMPERATURE", "0.0")))
    arg_parser.add_argument("--api-key", type=str, default=os.getenv("CBORG_API_KEY"))
    arg_parser.add_argument("--base-url", type=str, default=os.getenv("CBORG_BASE_URL", "https://api.cborg.lbl.gov/v1"))
    arg_parser.add_argument("--max-tokens", type=int, default=None)
    arg_parser.add_argument("--top-p", type=float, default=None)
    
    args = arg_parser.parse_args()

    if not args.api_key:
        args.api_key = getpass.getpass("Enter your CBorg API key: ")

    llm = init_cborg_chat_model(
        model=args.model,
        temperature=args.temperature,
        api_key=args.api_key,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )

    start_time = time.time()
    try:
        asyncio.run(main(llm))
    except Exception:
        logging.exception("Fatal error in run_pipeline")
        sys.exit(1)
    finally:
        logging.info(f"=== PIPELINE FINISHED in {time.time() - start_time:.2f} seconds ===")