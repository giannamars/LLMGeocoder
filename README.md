# LLMGeocoder

A Python pipeline for extracting structured geographic and metadata information from PubMed literature using Large Language Models.

## Overview

LLMGeocoder automates the systematic review process by:

1. **Fetching** articles from PubMed based on search queries
2. **Extracting** structured information using LLMs (study type, locations, dates)
3. **Geocoding** extracted locations using Nominatim/OpenStreetMap
4. **Identifying** genome accession numbers from multiple databases
5. **Exporting** results to CSV for further analysis

## Features

| | Feature | Description |
|---|---------|-------------|
| 🔬 | **Smart Classification** | Distinguishes human cases, animal cases, environmental detections, and excludes lab studies |
| 🌍 | **Hierarchical Geocoding** | Falls back through location specificity levels for best coordinate match |
| 🧬 | **Comprehensive Accession Extraction** | Supports NCBI, ENA, DDBJ, UniProt, GISAID, and more |
| 📄 | **Full-text Support** | Retrieves PMC full-text when available, falls back to abstracts |
| 🔄 | **Resumable** | Persists progress to pickle files for interrupted runs |
| ⚡ | **Async Processing** | Concurrent processing with rate limiting for CBORG API compliance |
