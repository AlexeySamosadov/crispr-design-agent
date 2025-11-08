# Structural Features

This document describes the structural featurization pipeline for incorporating 3D protein structure information into the CRISPR design models.

## Overview

The structural features module extracts contact maps, distance matrices, and graph representations from PDB and AlphaFold structure files. These features can be combined with sequence embeddings in a multimodal model to improve prediction accuracy.

## Components

### 1. Feature Extraction (`src/crispr_design_agent/features/structural.py`)

#### ContactMapExtractor
Extracts distance matrices and binary contact maps from PDB files.

**Key parameters:**
- `distance_threshold`: Distance cutoff in Angstroms (default: 8.0Å)
- `use_ca_only`: Use C-alpha atoms only vs. all heavy atoms

**Methods:**
- `extract_from_pdb()`: Process experimental PDB structures
- `extract_from_alphafold()`: Process AlphaFold predictions with pLDDT confidence filtering

#### GraphFeatureExtractor
Converts contact maps into graph neural network representations.

**Features generated:**
- Edge indices and attributes (distances + derived features)
- Local geometric features (k-nearest neighbor statistics)

### 2. Extraction Script (`scripts/extract_structure_features.py`)

Command-line tool for batch processing PDB files.

**Usage examples:**

```bash
# Extract features from AlphaFold structures
python scripts/extract_structure_features.py \
  --pdb-dir data/structures/alphafold \
  --output-dir features/structures \
  --is-alphafold \
  --distance-threshold 8.0 \
  --max-workers 4

# Extract features from experimental PDB files
python scripts/extract_structure_features.py \
  --pdb-dir data/structures/pdb \
  --output-dir features/structures \
  --pattern "*.pdb" \
  --limit 100

# Generate processing manifest
python scripts/extract_structure_features.py \
  --pdb-dir data/structures/alphafold \
  --manifest-output features/manifest.csv
```

**Output format:**
Each structure generates a `.pt` file containing:
- `distance_matrix`: [N, N] pairwise residue distances
- `contact_map`: [N, N] binary contact map
- `edge_index`: [2, E] graph edge connectivity
- `edge_attr`: [E, 3] edge features
- `local_geometry`: [N, 5] local geometric features
- `confidence`: [N] pLDDT scores (AlphaFold only)
- `residue_ids`: List of residue identifiers

### 3. Multimodal Training Module (`src/crispr_design_agent/training/multimodal_module.py`)

#### StructuralEncoder
Graph neural network that processes structural features.

**Architecture:**
- Edge encoder: Embeds distance features
- Geometric encoder: Processes local geometry
- GNN layers: Message passing over contact graph
- Output projection: Final structural representation

#### MultimodalLightningModule
Combines sequence and structural encoders with fusion layer.

**Configuration:**
```yaml
model:
  use_structure: true
  structure:
    contact_embedding_dim: 64
    hidden_dim: 128
    num_layers: 3
    dropout: 0.1
```

## Data Requirements

### AlphaFold Structures
Download from [AlphaFold DB](https://alphafold.ebi.ac.uk/):
- Store in `data/structures/alphafold/`
- Contains pLDDT confidence scores in B-factor field
- Use `--is-alphafold` flag when extracting

### Experimental PDB Structures
Download from [RCSB PDB](https://www.rcsb.org/):
- Store in `data/structures/pdb/`
- Extract without AlphaFold-specific processing

## Training Workflow

1. **Extract structural features:**
   ```bash
   python scripts/extract_structure_features.py \
     --pdb-dir data/structures/alphafold \
     --output-dir features/structures \
     --is-alphafold
   ```

2. **Train multimodal model:**
   ```bash
   python scripts/train_multitask.py \
     --config configs/model/multimodal.yaml \
     --limit 10000
   ```

3. **Monitor training:**
   - Checkpoints: `models/checkpoints/`
   - Logs: `logs/multitask/`

## Performance Considerations

- **Memory:** Structural features add ~5-10MB per protein
- **Compute:** GNN processing adds ~30% training time
- **Parallelization:** Use `--max-workers` for extraction

## Contact Definition

Two residues are in contact if:
- **C-alpha distance** ≤ 8.0Å (default)
- **Heavy atom distance** ≤ 4.5Å (alternative)

Adjust via `--distance-threshold` parameter.

## AlphaFold Confidence Filtering

Low-confidence regions (pLDDT < 70) are masked:
- Contacts to/from these residues are removed
- Prevents unreliable structural information

Adjust threshold in `extract_from_alphafold()`.

## Troubleshooting

### Missing residues
PDB files may have missing residues. The extractor:
- Skips non-standard residues
- Marks missing distances as NaN
- Logs warnings for incomplete structures

### Large structures
For proteins > 2000 residues:
- Consider domain splitting
- Use lower `max_length` in config
- Enable gradient checkpointing

### Memory errors during extraction
- Reduce `--max-workers`
- Process in batches with `--limit`
- Use C-alpha only mode

## References

- [AlphaFold Database](https://alphafold.ebi.ac.uk/)
- [RCSB PDB](https://www.rcsb.org/)
- [Biopython PDB module](https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ)
