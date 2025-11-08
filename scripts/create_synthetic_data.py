#!/usr/bin/env python3
"""Create synthetic datasets for quick testing."""

import random
import pandas as pd
import pathlib

# Random seed for reproducibility
random.seed(42)

# Amino acids
AAs = list("ACDEFGHIKLMNPQRSTVWY")

def random_sequence(length=50):
    """Generate random protein sequence."""
    return ''.join(random.choices(AAs, k=length))

def create_dms_data(n=1000):
    """Create synthetic DMS (Deep Mutational Scanning) data."""
    data = []
    for i in range(n):
        seq = random_sequence(random.randint(30, 100))
        effect = random.gauss(0, 1.5)  # Effect score
        data.append({
            'sequence': seq,
            'effect': effect,
            'protein_id': f'PROT{i % 100}',
            'position': random.randint(1, len(seq)),
        })
    return pd.DataFrame(data)

def create_clinvar_data(n=1000):
    """Create synthetic ClinVar pathogenicity data."""
    data = []
    for i in range(n):
        seq = random_sequence(random.randint(20, 80))
        # Binary pathogenicity
        is_pathogenic = random.choice([0, 1])
        data.append({
            'sequence': seq,
            'is_pathogenic': float(is_pathogenic),
            'variant_id': f'VAR{i}',
            'gene': f'GENE{i % 50}',
        })
    return pd.DataFrame(data)

def create_depmap_data(n=1000):
    """Create synthetic DepMap gene effect data."""
    data = []
    for i in range(n):
        seq = random_sequence(random.randint(40, 120))
        gene_effect = random.gauss(-0.5, 0.8)  # Negative = essential
        data.append({
            'sequence': seq,
            'gene_effect': gene_effect,
            'gene_name': f'GENE{i % 100}',
            'cell_line': f'CELL{i % 20}',
        })
    return pd.DataFrame(data)

def create_uniprot_data(n=500):
    """Create synthetic UniProt sequence data."""
    data = []
    for i in range(n):
        seq = random_sequence(random.randint(50, 200))
        data.append({
            'sequence': seq,
            'uniprot_id': f'P{i:05d}',
            'protein_name': f'Protein_{i}',
            'organism': 'Homo sapiens' if i % 2 == 0 else 'Mus musculus',
        })
    return pd.DataFrame(data)

def main():
    output_dir = pathlib.Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating synthetic datasets...")

    # Create datasets
    dms_df = create_dms_data(1000)
    print(f"✓ DMS: {len(dms_df)} samples")
    dms_df.to_parquet(output_dir / 'dms.parquet', index=False)

    clinvar_df = create_clinvar_data(1000)
    print(f"✓ ClinVar: {len(clinvar_df)} samples")
    clinvar_df.to_parquet(output_dir / 'clinvar.parquet', index=False)

    depmap_df = create_depmap_data(1000)
    print(f"✓ DepMap: {len(depmap_df)} samples")
    depmap_df.to_parquet(output_dir / 'depmap.parquet', index=False)

    uniprot_df = create_uniprot_data(500)
    print(f"✓ UniProt: {len(uniprot_df)} sequences")
    uniprot_df.to_parquet(output_dir / 'uniprot_sequences.parquet', index=False)

    print(f"\n✅ All synthetic datasets created in {output_dir}/")
    print("\nDataset Statistics:")
    print(f"  - dms.parquet: {len(dms_df)} rows")
    print(f"  - clinvar.parquet: {len(clinvar_df)} rows")
    print(f"  - depmap.parquet: {len(depmap_df)} rows")
    print(f"  - uniprot_sequences.parquet: {len(uniprot_df)} rows")
    print("\nYou can now test the training pipeline with:")
    print("  python scripts/train_multitask.py --limit 500 --experiment-name 'synthetic-test'")

if __name__ == '__main__':
    main()
