# src/simple_nodes.py
import asyncio
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
            pass

from .embedding_retrieval import retrieve_top_chunks, build_index
from .utils import _extract_accessions_from_text, _extract_accessions_categorized


class LocationInfo(BaseModel):
    """One occurrence of a melioidosis case / environmental detection."""
    
    region: str = Field(description="Geographic region or 'unknown'.")
    country: str = Field(description="Full country name or 'unknown'.")
    location: str = Field(description="Specific place or 'unknown'")
    amenity: str = Field(default="unknown")
    street: str = Field(default="unknown")
    city: str = Field(default="unknown")
    county: str = Field(default="unknown")
    state: str = Field(default="unknown")
    postalcode: str = Field(default="unknown")

    @root_validator(pre=True)
    def _fill_unknown(cls, values):
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
    accession_categories: Optional[Dict[str, List[str]]]


class StudyInfo(BaseModel):
    """The top-level JSON object that the LLM must return."""
    
    study_type: str = Field(
        description='One of "Human cases", "Animal cases", "Environmental cases", "Excluded", or "unknown"'
    )
    sample_date: str = Field(description="Four-digit year or 'unknown'")
    location: List[LocationInfo] = Field(description="One or more occurrence objects")

    @validator("study_type")
    def _check_study_type(cls, v: str) -> str:
        allowed = {"Human cases", "Animal cases", "Environmental cases", "Excluded", "unknown"}
        if v not in allowed:
            raise ValueError(f'study_type must be one of {allowed}; got "{v}"')
        return v

    @validator("sample_date", pre=True)
    def _normalize_sample_date(cls, v) -> str:
        if not v or (isinstance(v, str) and not v.strip()):
            return "unknown"
        v = str(v).strip()
        if v.lower() == "unknown":
            return "unknown"
        if not re.fullmatch(r"\d{4}", v):
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
    max_retry: int = 10,
) -> Dict[str, Any]:
    """
    Sends the document to the LLM with retry logic.
    Based on CBorg API recommendations.
    """
    formatted_prompt = f"PMID: {pmid}\n\n" + prompt.format(full_document=doc_text)

    messages = [
        SystemMessage(
            content=(
                "You are a JSON-only API. Output raw JSON with no markdown formatting, "
                "no code fences, no explanations. Start with { and end with }."
            )
        ),
        HumanMessage(content=formatted_prompt),
    ]

    n = 0
    while True:
        n += 1
        
        try:
            response = await llm.ainvoke(messages)
            
            # Success - parse the response
            raw_content = response.content.strip()

            # Strip markdown code fences
            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_content)
            if fence_match:
                raw_content = fence_match.group(1).strip()

            # Extract JSON object
            json_match = re.search(r"\{[\s\S]*\}", raw_content)
            if json_match:
                raw_content = json_match.group(0)

            try:
                parsed_obj = parser.parse(raw_content)
                return parsed_obj.dict()
            except OutputParserException as exc:
                logging.warning(f"⚠️ LLM output could not be parsed for PMID {pmid}: {exc}")
                logging.debug(f"Raw response:\n{response.content[:500]}")
                return {"study_type": "unknown", "sample_date": "unknown", "location": []}

        except Exception as e:
            if n < max_retry:
                wait_time = max(1, n * 5)
                logging.warning(f"CBORG API ERROR (PMID {pmid}, Attempt {n}/{max_retry}): {e}")
                logging.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logging.error(f"CBORG API FAILED after {max_retry} attempts for PMID {pmid}: {e}")
                return {"study_type": "unknown", "sample_date": "unknown", "location": []}


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

    # Use full text for shorter documents, embeddings for longer
    if len(full_text) <= 5000:
        text_for_llm = full_text
    else:
        excerpt = retrieve_top_chunks(pmid, top_k=5)
        if excerpt is None:
            logging.info(f"Building vector store for PMID {pmid} (first use).")
            success = build_index(pmid, full_text)
            if not success:
                logging.error(f"Failed to build vector store for PMID {pmid}; using full text.")
                text_for_llm = full_text
            else:
                excerpt = retrieve_top_chunks(pmid, top_k=5)
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
    else:
        meta = state.get("doc_metadata", {})
        meta_year = _year_from_metadata(meta)
        parsed["sample_date"] = meta_year if meta_year else "unknown"

    new_state = dict(state)
    new_state["answer"] = parsed
    return new_state


async def detect_accession_numbers(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract accession numbers from the document text (categorized)."""
    raw_text = state.get("full_document", "")
    state["accession_numbers"] = _extract_accessions_from_text(raw_text)
    state["accession_categories"] = _extract_accessions_categorized(raw_text)
    return state