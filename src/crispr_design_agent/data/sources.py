"""Definitions of canonical datasets used by the CRISPR design agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class DatasetSource:
    """Metadata holder describing a dataset and how to access it."""

    name: str
    short_name: str
    description: str
    details_url: str
    download_url: Optional[str]
    license: str
    requires_auth: bool = False
    notes: str = ""
    file_hint: str = ""
    citation: str = ""
    filters: Dict[str, str] = field(default_factory=dict)


def _dataset(name: str, **kwargs) -> DatasetSource:
    return DatasetSource(name=name, **kwargs)


AVAILABLE_SOURCES: Dict[str, DatasetSource] = {
    "mavedb": _dataset(
        name="mavedb",
        short_name="MaveDB",
        description="Deep mutational scanning score sets capturing the functional impact of amino-acid substitutions.",
        details_url="https://www.mavedb.org/",
        download_url="https://www.mavedb.org/api/score-sets",
        license="CC BY 4.0",
        file_hint="JSON manifest with links to TSV score-set files.",
        citation="Esposito et al. 2019, MaveDB: an open-source platform to distribute and interpret data from MAVE experiments.",
        notes="Full downloads can be large; filter by assay or gene via the API parameters.",
    ),
    "uniprot_sprot": _dataset(
        name="uniprot_sprot",
        short_name="UniProtKB/Swiss-Prot",
        description="Manually curated protein sequences with functional annotations and cross-references.",
        details_url="https://www.uniprot.org/help/downloads",
        download_url="https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%28reviewed:true%29",
        license="Creative Commons Attribution 4.0",
        file_hint="Compressed FASTA stream of reviewed sequences.",
        notes="Use the REST query parameters to narrow down to taxa or proteins of interest.",
        citation="UniProt Consortium, Nucleic Acids Research 2023.",
    ),
    "alphafold": _dataset(
        name="alphafold",
        short_name="AlphaFold DB",
        description="Predicted 3D structures for proteins with per-residue confidence scores.",
        details_url="https://alphafold.ebi.ac.uk/download",
        download_url="https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v4.tar",
        license="CC BY 4.0",
        file_hint="Tar archives grouped by proteomes; default uses the human proteome.",
        notes="Replace the proteome archive with the organism(s) you need.",
        citation="Varadi et al. 2022. AlphaFold Protein Structure Database.",
    ),
    "depmap": _dataset(
        name="depmap",
        short_name="DepMap CRISPR",
        description="Genome-wide CRISPR knockout screens with gene effect scores across cell lines.",
        details_url="https://depmap.org/portal/download/",
        download_url=None,
        license="DepMap Open Data - CC BY 4.0",
        requires_auth=True,
        notes="Requires accepting the DepMap terms and generating a personal token; download manually then place files under data/raw/depmap.",
        citation="Tsherniak et al. 2017; DepMap Public 24Q1 release.",
    ),
    "clinvar": _dataset(
        name="clinvar",
        short_name="ClinVar",
        description="Clinically characterized variants with pathogenicity assertions.",
        details_url="https://www.ncbi.nlm.nih.gov/clinvar/",
        download_url="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz",
        license="Public Domain",
        file_hint="Tab-delimited summary with significance labels.",
        citation="Landrum et al. ClinVar: public archive of relationships among sequence variation and human phenotype.",
    ),
    "depmap_expression": _dataset(
        name="depmap_expression",
        short_name="DepMap Expression",
        description="RNA-seq TPM data matched to DepMap cell lines.",
        details_url="https://depmap.org/portal/download/all/",
        download_url=None,
        license="DepMap Open Data - CC BY 4.0",
        requires_auth=True,
        notes="Download alongside DepMap gene effect scores; place under data/raw/depmap.",
    ),
}


def list_available_sources() -> Dict[str, DatasetSource]:
    """Return dict of dataset key -> metadata."""
    return AVAILABLE_SOURCES


def get_source(key: str) -> DatasetSource:
    """Look up dataset metadata by registry key."""
    try:
        return AVAILABLE_SOURCES[key]
    except KeyError as exc:  # pragma: no cover - guard rails
        raise KeyError(f"Unknown dataset key '{key}'. Known keys: {list(AVAILABLE_SOURCES)}") from exc
