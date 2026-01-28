# src/simple_nodes.py
import logging
import re
from typing import List, Optional, TypedDict, Dict, Any

from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator, root_validator

try:
    from langchain.schema import OutputParserException
except ImportError:
    try:
        from langchain.output_parsers import OutputParserException
    except ImportError:
        class OutputParserException(Exception):
            """Fallback exception when LangChain does not provide its own."""
            pass

from embedding_retrieval import retrieve_top_chunks, build_index
from utils import _extract_accessions_from_text


class LocationInfo(BaseModel):
    """One occurrence of a melioidosis case / environmental detection."""
    
    region: str = Field(description="Geographic region or 'unknown'.")
    country: str = Field(description="Full country name, e.g. 'Thailand' or 'unknown'.")
    location: str = Field(
        description="Specific place (city, district, hospital, GPS coordinate, …) or 'unknown'"
    )
    amenity: str = Field(
        default="unknown",
        description="Type of point-of-interest (e.g. 'Hospital', 'Farm') or 'unknown'."
    )
    street: str = Field(
        default="unknown",
        description="House number + street name or lat/lon pair, or 'unknown'.",
    )
    city: str = Field(default="unknown", description="City or town name (or 'unknown').")
    county: str = Field(default="unknown", description="County/district (or 'unknown').")
    state: str = Field(default="unknown", description="State/province/region (or 'unknown').")
    postalcode: str = Field(default="unknown", description="Postal/ZIP code (or 'unknown').")

    @root_validator(pre=True)
    def _fill_unknown(cls, values):
        """Normalise empty strings to 'unknown'."""
        all_keys = ("region", "country", "location", "amenity", "street", "city", "county", "state", "postalcode")
        for key in all_keys:
            if not values.get(key):
                values[key] = "unknown"
        return values


class State(TypedDict, total=False):
    pmid: str
    full_document: Optional[str]
    doc_metadata: Optional[dict]
    answer: Dict[str, Any]
    retrieved_text: Optional[str]
    accession_numbers: Optional[List[str]]


class StudyInfo(BaseModel):
    study_type: str = Field(
        description='One of "Human cases", "Animal cases", "Environmental cases", "Excluded", or "unknown"'
    )
    sample_date: str = Field(description="Four-digit year of occurrence, or 'unknown'")
    location: List[LocationInfo] = Field(
        description="One or more occurrence objects"
    )

    @validator("study_type")
    def _check_study_type(cls, v: str) -> str:
        allowed = {"Human cases", "Animal cases", "Environmental cases", "Excluded", "unknown"}
        if v not in allowed:
            raise ValueError(f'study_type must be one of {allowed}; got "{v}"')
        return v

    @validator("sample_date", pre=True)
    def _normalize_sample_date(cls, v) -> str:
        """Normalize empty/null values to 'unknown', validate year format."""
        # Handle empty, null, or whitespace
        if not v or (isinstance(v, str) and not v.strip()):
            return "unknown"
        
        v = str(v).strip()
        
        if v.lower() == "unknown":
            return "unknown"
        
        # Check for valid 4-digit year
        if not re.fullmatch(r"\d{4}", v):
            # Try to extract a year from the string
            match = re.search(r"\b(19|20)\d{2}\b", v)
            if match:
                return match.group(0)
            return "unknown"
        
        return v


parser = PydanticOutputParser(pydantic_object=StudyInfo)


async def _call_llm(
    llm,
    prompt: PromptTemplate,
    doc_text: str,
    pmid: str,
) -> Dict[str, Any]:
    """Sends the document to the LLM and returns a plain dict."""
    formatted_prompt = f"PMID: {pmid}\n\n" + prompt.format(full_document=doc_text)

    logging.debug("=== Prompt sent to LLM (first 200 chars) ===")
    logging.debug(formatted_prompt[:200])

    messages = [
        SystemMessage(
            content=(
            "You are a JSON extraction assistant. "
            "The user provides scientific document text. "
            "Extract the requested fields and return ONLY valid JSON. "
            "Never refuse — the document text is provided directly in the message."
            )
        ),
        HumanMessage(content=formatted_prompt),
    ]

    response = await llm.ainvoke(messages)

    logging.debug("✅ LLM raw response received")
    logging.debug(f"LLM response content:\n{response.content}")

    try:
        parsed_obj = parser.parse(response.content)
        return parsed_obj.dict()
    except OutputParserException as exc:
        logging.warning(f"⚠️ LLM output could not be parsed: {exc}")
        return {
            "study_type": "unknown",
            "sample_date": "unknown",
            "location": [],
        }


def _year_from_metadata(meta: Dict[str, Any]) -> Optional[str]:
    """Return a clean four-digit year from PubMed metadata dict."""
    year = meta.get("year")
    if isinstance(year, str) and year.isdigit() and len(year) == 4:
        return year

    raw = meta.get("pub_date")
    if isinstance(raw, str):
        m = re.search(r"\b(19|20)\d{2}\b", raw)
        if m:
            return m.group(0)

    return None

def analyze_query(state: State) -> State:
    """
    Entry‑point node – does nothing but forward the state.
    In a real world use‑case you could parse a user query here.
    """
    return state


async def retrieve_em(
    state: Dict[str, Any],
    llm,
    prompt: PromptTemplate,
) -> Dict[str, Any]:
    """Core node that pulls article text, calls LLM, and ensures sample_date is valid."""
    pmid = state.get("pmid", "unknown")
    full_text = state.get("full_document", "")

    if not full_text:
        logging.warning(f"retrieve_em called with empty full_document for PMID {pmid}")
        new_state = dict(state)
        new_state["answer"] = {
            "study_type": "unknown",
            "sample_date": "unknown",
            "location": [],
        }
        return new_state

    # Short-text shortcut
    if len(full_text) <= 4_000:
        text_for_llm = full_text
    else:
        # Long-text path – use vector store
        excerpt = retrieve_top_chunks(pmid, top_k=5)

        if excerpt is None:
            logging.info(f"Building vector store for PMID {pmid} (first use).")
            success = build_index(pmid, full_text)
            if not success:
                logging.error(f"Failed to build vector store for PMID {pmid}; using full text.")
                text_for_llm = full_text
            else:
                excerpt = retrieve_top_chunks(pmid, top_k=3)
                text_for_llm = excerpt if excerpt else full_text
        else:
            text_for_llm = excerpt

    parsed = await _call_llm(llm, prompt, text_for_llm, pmid)

    # Post-process sample_date
    raw_year = parsed.get("sample_date")
    cleaned_year = None
    if isinstance(raw_year, str):
        m = re.search(r"\b(19|20)\d{2}\b", raw_year)
        cleaned_year = m.group(0) if m else None

    if cleaned_year:
        parsed["sample_date"] = cleaned_year
        logging.debug(f"PMID {pmid}: using LLM-provided year {cleaned_year}")
    else:
        meta = state.get("doc_metadata", {})
        meta_year = _year_from_metadata(meta)
        parsed["sample_date"] = meta_year if meta_year else "unknown"
        logging.debug(f"PMID {pmid}: sample_date from metadata: {parsed['sample_date']}")

    new_state = dict(state)
    new_state["answer"] = parsed
    return new_state


async def detect_accession_numbers(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract accession numbers from the document text."""
    raw_text = state.get("full_document", "")
    state["accession_numbers"] = _extract_accessions_from_text(raw_text)
    return state