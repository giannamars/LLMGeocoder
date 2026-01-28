# src/pubmed_loader.py
"""
RobustPubMedLoader – a LangChain BaseLoader that fetches PubMed records
(abstracts or full‑text from PubMed Central) and persists the set of
already‑processed PMIDs so a long run can be resumed.

"""

import os
import pickle
import time
import logging
import xml.etree.ElementTree as ET
from io import BytesIO, StringIO

from typing import List, Optional, Dict, Set, Any

import http.client
import xml.parsers.expat
from Bio import Entrez, Medline

import re

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader



# ----------------------------------------------------------------------
# Helper – turn a PubMed “DP” (date of publication) string into a 4‑digit year
# ----------------------------------------------------------------------
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")   # matches 1900‑2099

def _extract_year_from_dp(dp: str | None) -> str | None:
    """
    PubMed stores the publication date in the ``DP`` field, e.g.
        "2021 Jan  "2020"
        "1999 Dec"
    This helper pulls the first four‑digit year it finds, or returns ``None``.
    """
    if not dp:
        return None
    m = _YEAR_RE.search(dp)
    return m.group(0) if m else None

# ----------------------------------------------------------------------
# Global Entrez configuration – read once from the environment.
# ----------------------------------------------------------------------
Entrez.email = os.getenv(
    "ENTREZ_EMAIL", "glmarschmann@lbl.gov"
)  # NCBI requires a real e‑mail address.
Entrez.api_key = os.getenv("NCBI_API_KEY") or None
Entrez.tool = "LangChainPubMed"


class RobustPubMedLoader(BaseLoader):
    """
    Retrieve PubMed entries (abstracts or full‑text) for a given query.
    The class keeps a persistent set of processed PMIDs so that a
    partially‑finished run can be resumed safely.

    Parameters
    ----------
    query : str
        PubMed search string (e.g. ``"Burkholderia AND pseudomallei"``).
    max_docs : int, optional
        Upper bound on the number of PMIDs to retrieve in this call.
    batch_size : int, optional
        Number of PMIDs fetched from Entrez in a single ``efetch`` call.
    api_key : str | None, optional
        NCBI API key – overrides the ``NCBI_API_KEY`` environment variable.
    email : str | None, optional
        E‑mail address passed to NCBI – overrides the ``ENTREZ_EMAIL`` env‑var.
    tool : str, optional
        Identifier that appears in NCBI logs (default ``"LangChainPubMed"``).
    sleep_between : float, optional
        Seconds to ``time.sleep`` between consecutive Entrez calls
        (respect NCBI’s ~3 req/s limit).
    retries : int, optional
        Number of exponential‑back‑off retries for transient network errors.
    fetch_full_text : bool, optional
        If ``True`` attempt to retrieve the full‑text from PMC; fall back
        to abstract when not available.
    processed_pmids_file : str, optional
        Path to a pickle that stores ``{\"pmids\": set(), \"results\": []}``.
        The file is created automatically if missing.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        query: str,
        max_docs: int = 100,
        batch_size: int = 200,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        tool: str = "LangChainPubMed",
        sleep_between: float = 0.3,
        retries: int = 5,
        fetch_full_text: bool = True,
        processed_pmids_file: str = "processed_pmids.pkl",
    ) -> None:
        self.query = query
        self.max_docs = max_docs
        self.batch_size = batch_size
        self.sleep_between = sleep_between
        self.retries = retries
        self.fetch_full_text = fetch_full_text
        self.tool = tool

        # ------------------------------------------------------------------
        # Apply any overrides for Entrez configuration
        # ------------------------------------------------------------------
        if api_key:
            Entrez.api_key = api_key
        if email:
            Entrez.email = email
        Entrez.tool = tool

        # ------------------------------------------------------------------
        # Persistence handling – load or initialise the pickle that tracks PMIDs
        # ------------------------------------------------------------------
        self.processed_pmids_file = processed_pmids_file
        self.processed_data: Dict[str, Set[str]] = self._load_processed_pickle()
        self.processed_pmids: Set[str] = self.processed_data["pmids"]

    # ------------------------------------------------------------------
    # Helper – load processed‑PMIDs pickle (private)
    # ------------------------------------------------------------------
    def _load_processed_pickle(self) -> Dict[str, Set[str]]:
        """
        Return the dict stored in ``processed_pmids_file``.
        Expected shape: ``{\"pmids\": set(), \"results\": []}``.
        If the file does not exist or is corrupt a fresh dict is returned.
        """
        if not os.path.isfile(self.processed_pmids_file):
            return {"pmids": set(), "results": []}
        try:
            with open(self.processed_pmids_file, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "pmids" in data and isinstance(data["pmids"], set):
                return data
        except Exception as exc:  # pragma: no cover
            logging.warning(
                f"Could not read processed‑PMIDs pickle {self.processed_pmids_file}: {exc}"
            )
        return {"pmids": set(), "results": []}

    # ------------------------------------------------------------------
    # Helper – write the processed‑PMIDs pickle (private)
    # ------------------------------------------------------------------
    def _save_processed_pickle(self) -> None:
        """Write the in‑memory ``processed_data`` back to disk."""
        try:
            with open(self.processed_pmids_file, "wb") as f:
                pickle.dump(self.processed_data, f)
        except Exception as exc:  # pragma: no cover
            logging.error(
                f"Failed to write processed‑PMIDs pickle {self.processed_pmids_file}: {exc}"
            )

    # ------------------------------------------------------------------
    # Static helper – make sure an Entrez response is a plain string
    # ------------------------------------------------------------------
    @staticmethod
    def _to_str(data: Any) -> str:
        """
        Convert the object returned by ``Entrez.efetch`` (bytes or str) into a
        plain Python string.  Errors are ignored so that malformed Unicode does
        not break the pipeline.
        """
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")
        return str(data)

    # ------------------------------------------------------------------
    # Public API required by ``BaseLoader``
    # ------------------------------------------------------------------
    def load(self) -> List[Document]:
        """
        Fetch up to ``max_docs`` new PubMed records that have not been processed
        yet, and return them as a list of ``Document`` objects.
        """
        pmids = self._get_pmids()
        if not pmids:
            logging.info("No new PMIDs to fetch.")
            return []

        documents = self._fetch_documents(pmids)

        # ``max_docs`` is a hard cap – slice in case the batch returned more.
        return documents[: self.max_docs]

    # ------------------------------------------------------------------
    # Step 1 – retrieve a list of PMIDs that match the query
    # ------------------------------------------------------------------
    def _get_pmids(self) -> List[str]:
        """
        Run ``Entrez.esearch`` for the supplied query and return the list of
        PMIDs that have **not** been processed yet.
        """
        logging.info(
            f"Searching PubMed for up to {self.max_docs} PMIDs (query: {self.query})"
        )
        with Entrez.esearch(
            db="pubmed", term=self.query, retmax=self.max_docs
        ) as handle:
            results = Entrez.read(handle)

        all_pmids = results.get("IdList", [])
        new_pmids = [pmid for pmid in all_pmids if pmid not in self.processed_pmids]

        logging.info(f"Entrez returned {len(all_pmids)} PMIDs – {len(new_pmids)} are new.")
        return new_pmids

    # ------------------------------------------------------------------
    # Step 2 – map PMIDs → PMCIDs (needed for full‑text)
    # ------------------------------------------------------------------
    def _get_pmc_ids(self, pmid_list: List[str]) -> Dict[str, str]:
        """
        Resolve a batch of PMIDs to their corresponding PMCIDs using
        ``Entrez.elink``.  Returns a dict ``{pmid: pmcid}``.  The function
        respects ``self.retries`` and ``self.sleep_between`` for back‑off.
        """
        pmc_map: Dict[str, str] = {}
        for i in range(0, len(pmid_list), 200):
            chunk = pmid_list[i : i + 200]
            attempt = 0
            delay = 1.0
            while attempt < self.retries:
                try:
                    with Entrez.elink(
                        dbfrom="pubmed",
                        db="pmc",
                        id=chunk,
                        linkname="pubmed_pmc",
                    ) as handle:
                        raw_xml = handle.read()
                    # ``Entrez.read`` expects a file‑like object.
                    handle_parsed = BytesIO(raw_xml)
                    results = Entrez.read(handle_parsed)

                    for entry in results:
                        if "LinkSetDb" not in entry or not entry["LinkSetDb"]:
                            continue
                        pmid = entry["IdList"][0]
                        pmcid = entry["LinkSetDb"][0]["Link"][0]["Id"]
                        pmc_map[pmid] = pmcid
                    break  # success → exit retry loop
                except (http.client.IncompleteRead, xml.parsers.expat.ExpatError, ValueError) as err:
                    attempt += 1
                    logging.warning(
                        f"elink retry {attempt}/{self.retries} for chunk {chunk} failed: {err}"
                    )
                    time.sleep(delay)
                    delay *= 2  # exponential back‑off
                except Exception as err:  # pragma: no cover
                    logging.error(f"Unexpected elink error for chunk {chunk}: {err}")
                    raise
            else:
                logging.error(f"Failed to fetch PMC IDs after {self.retries} attempts for {chunk}")
            time.sleep(self.sleep_between)
        return pmc_map

    # ------------------------------------------------------------------
    # Step 3 – fetch full‑text XML from PMC and extract plain text
    # ------------------------------------------------------------------
    def _extract_pmc_full_text(self, pmcid: str) -> Optional[str]:
        """
        Retrieve the full‑text article XML for a PMCID and return the
        concatenated body paragraphs as a single string.  Returns ``None``
        if the article cannot be fetched or contains no body text.
        """
        for attempt in range(self.retries):
            try:
                with Entrez.efetch(
                    db="pmc", id=pmcid, rettype="full", retmode="xml"
                ) as handle:
                    xml_data = handle.read()
                root = ET.fromstring(xml_data)

                paragraphs = [
                    "".join(p.itertext()).strip()
                    for p in root.findall(".//body//p")
                    if "".join(p.itertext()).strip()
                ]

                if paragraphs:
                    return "\n\n".join(paragraphs)
                return None
            except Exception as exc:  # pragma: no cover
                logging.warning(
                    f"PMC fetch failed for PMCID {pmcid} (attempt {attempt+1}/{self.retries}): {exc}"
                )
                time.sleep(2 ** attempt)
        return None

    # ------------------------------------------------------------------
    # Generic helper – fetch from any Entrez DB with exponential back‑off
    # ------------------------------------------------------------------
    def _fetch_with_backoff(
        self,
        db: str,
        ids: List[str],
        rettype: str = "medline",
        retmode: str = "text",
    ) -> bytes:
        """
        ``Entrez.efetch`` with retry logic.  Returns the raw response as
        ``bytes`` (the caller may decode it if needed).
        """
        for attempt in range(self.retries):
            try:
                with Entrez.efetch(db=db, id=ids, rettype=rettype, retmode=retmode) as handle:
                    return handle.read()
            except Exception as exc:  # pragma: no cover
                logging.warning(
                    f"Entrez efetch ({db}) failed on attempt {attempt+1}/{self.retries}: {exc}"
                )
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to fetch from {db} after {self.retries} attempts")

    # ------------------------------------------------------------------
    # Step 4 – build Document objects from the retrieved records
    # ------------------------------------------------------------------

    _YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")   # matches 1900‑2099

    def _extract_year_from_dp(dp: Optional[str]) -> Optional[str]:
        """
        PubMed stores the publication date in the ``DP`` field (e.g.
        "2025 Aug 23", "2025 Jul‑Sep", "2025").
        This helper returns the first four‑digit year it finds, or ``None``.
        """
        if not dp:
            return None
        m = _YEAR_RE.search(dp)
        return m.group(0) if m else None

    def _fetch_documents(self, pmids: List[str]) -> List[Document]:
        """
        For each PMID in ``pmids`` retrieve either the full‑text (if
        ``fetch_full_text`` is True) or the abstract + title.  Records are
        turned into ``Document`` objects and added to the processed‑PMID set.
        """
        documents: List[Document] = []

        # Resolve PMCIDs only once (if we need full‑text)
        pmc_map: Dict[str, str] = (
            self._get_pmc_ids(pmids) if self.fetch_full_text else {}
        )

        # Process PMIDs in batches (respect NCBI limits)
        for start in range(0, len(pmids), self.batch_size):
            chunk = pmids[start : start + self.batch_size]

            # ---------------------------------------------------------
            # 1️⃣  Pull the MEDLINE records (title, abstract, etc.)
            # ---------------------------------------------------------
            raw_medline = self._fetch_with_backoff("pubmed", chunk)
            raw_medline_str = self._to_str(raw_medline)  # convert to a plain string

            medline_records = list(
                Medline.parse(StringIO(raw_medline_str))
            )  # returns an iterator of dicts

            for record in medline_records:
                pmid = record.get("PMID")
                if not pmid:
                    logging.warning("Skipping record without PMID.")
                    continue

                # -----------------------------------------------------
                # Wrap each PMID's processing in a try/except so a single
                # failure does not abort the whole batch.
                # -----------------------------------------------------
                try:
                    # -------------------------------------------------
                    # Skip already‑processed PMIDs (safety net)
                    # -------------------------------------------------
                    if pmid in self.processed_pmids:
                        logging.info(f"Skipping already‑processed PMID {pmid}.")
                        continue

                    # -------------------------------------------------
                    # Assemble basic metadata
                    # -------------------------------------------------
                    raw_pub_date = record.get("DP", None)          # keep the raw DP string for reference
                    pub_year = _extract_year_from_dp(raw_pub_date)   # <-- NEW: clean 4‑digit year

                    metadata: Dict[str, Any] = {
                        "pmid": pmid,
                        "title": record.get("TI", ""),
                        "journal": record.get("JT", ""),
                        "authors": record.get("AU", []),
                        "affiliation": record.get("AD", None),
                        "pub_date": raw_pub_date,    # keep the raw string for reference
                        "year": pub_year, 
                        "source": "PubMed",  # may be overwritten if full‑text is found
                    }

                    # -------------------------------------------------
                    # 2️⃣  Try to fetch full‑text from PMC (if requested)
                    # -------------------------------------------------
                    content: Optional[str] = None
                    full_text_available = False

                    if self.fetch_full_text and pmid in pmc_map:
                        pmcid = pmc_map[pmid]
                        metadata["pmcid"] = pmcid
                        full_text = self._extract_pmc_full_text(pmcid)

                        # Detect the generic “PMCID is not available” placeholder.
                        if full_text and "PMCID is not available" not in full_text:
                            content = full_text
                            metadata["source"] = "PMC"
                            full_text_available = True
                            logging.info(
                                f"Full‑text retrieved for PMID {pmid} (PMCID {pmcid})."
                            )
                        else:
                            logging.debug(
                                f"PMCID {pmcid} for PMID {pmid} yielded no usable text."
                            )

                    # -------------------------------------------------
                    # 3️⃣  If we still have no content, fall back to abstract
                    # -------------------------------------------------
                    if not content:
                        logging.debug(f"PMID {pmid}: falling back to abstract.")
                        abstract_bytes = self._fetch_with_backoff(
                            "pubmed", [pmid], rettype="abstract", retmode="text"
                        )
                        abstract = self._to_str(abstract_bytes)

                        if abstract.strip():
                            title = metadata.get("title", "")
                            content = f"{title}\n\n{abstract}"
                            metadata["source"] = "PubMed"
                            logging.info(f"Abstract fallback used for PMID {pmid}.")
                        else:
                            logging.warning(
                                f"PMID {pmid} has no abstract either – document will be empty."
                            )

                    # -------------------------------------------------
                    # 4️⃣  Log the final length and decide whether to keep it
                    # -------------------------------------------------
                    txt_len = len(content.strip()) if content else 0
                    if txt_len == 0:
                        # Very rare – no full‑text and no abstract.
                        logging.warning(
                            f"PMID {pmid} resulted in EMPTY content (length=0). Skipping."
                        )
                        # Mark as processed so we don't keep trying.
                        self.processed_pmids.add(pmid)
                        continue
                    else:
                        logging.debug(f"PMID {pmid} → retrieved {txt_len} characters")

                    # -------------------------------------------------
                    # 5️⃣  Build the LangChain Document object
                    # -------------------------------------------------
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)

                    # -------------------------------------------------
                    # 6️⃣  Record that we have processed this PMID
                    # -------------------------------------------------
                    self.processed_pmids.add(pmid)

                    # Respect NCBI rate limits – ~3 requests per second max.
                    time.sleep(self.sleep_between)

                except Exception as exc:  # pragma: no cover
                    logging.error(f"Failed to fetch PMID {pmid}: {exc}")

        # -------------------------------------------------------------
        # Persist the updated PMID set (so a later run can resume)
        # -------------------------------------------------------------
        self.processed_data["pmids"] = self.processed_pmids
        self._save_processed_pickle()

        return documents
    
