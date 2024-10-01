# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:58:28 2024

@author: Luisa
"""

import pandas as pd
import requests
import time
from requests.exceptions import RequestException

# Function to retrieve the sequence from UniProt
def get_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    retries = 3  # Number of retries in case of failure
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)  # Set a timeout
            if response.status_code == 200:
                fasta_data = response.text
                sequence = ''.join(fasta_data.split('\n')[1:])
                return sequence
            else:
                return None
        except (RequestException, TimeoutError):
            print(f"Error retrieving {uniprot_id}. Attempt {attempt + 1}/{retries}")
            time.sleep(2)  # Wait before retrying

    return None  # Return None if all retries failed

# Function to extract peptide with flanking 7 amino acids around the phosphorylation site and make the site lowercase
# Also pad the peptide to 15 characters if necessary
def get_flanking_peptide(sequence, site_position):
    start_pos = max(0, site_position - 8)  # 7 amino acids upstream + 1 for the phosphorylation site itself
    end_pos = min(len(sequence), site_position + 7)  # 7 amino acids downstream
    
    flanking_peptide = sequence[start_pos:end_pos]  # Flanking region
    
    # Calculate the position of the phosphorylation site relative to the flanking peptide
    site_relative_pos = site_position - start_pos-1
    
    # Make the phosphorylation site lowercase at the exact position
    if site_relative_pos < len(flanking_peptide):
        flanking_peptide = (
            flanking_peptide[:site_relative_pos] + 
            flanking_peptide[site_relative_pos].lower()+'*' + 
            flanking_peptide[site_relative_pos+1:]
        )
    
    # If the peptide is less than 15 characters, pad it with underscores
    if len(flanking_peptide) < 15:
        flanking_peptide = flanking_peptide.ljust(15, '_')
    
    return flanking_peptide

# Function to process each phosphorylation site (pSites)
def process_psites(psites, sequence):
    peptides = []
    
    # Split the sites into separate entries based on comma
    sites = psites.split(', ')
    
    # For the first entry, we'll split by '_'
    first_site = sites[0]
    if '_' in first_site:
        res, pos = first_site.split('_')
        try:
            pos = int(pos[1:])  # Extract the numeric part (strip the first character)
            peptide = get_flanking_peptide(sequence, pos)
            peptides.append(f"{peptide}")
        except ValueError:
            peptides.append("Invalid position format")
    
    # For any remaining entries, treat them as part of the same protein, but only process the positions
    for second_site in sites[1:]:
        if second_site:
            try:
                second_pos = int(second_site[1:])  # Extract numeric part (e.g., Y187 -> 187)
                peptide = get_flanking_peptide(sequence, second_pos)
                peptides.append(f"{peptide}")
            except ValueError:
                peptides.append("Invalid position format")
        else:
            peptides.append("Invalid format")

    return ', '.join(peptides)

# Function to handle batching of UniProt requests
def process_in_batches(df, batch_size=5, delay=2):
    # Add a new column for the extracted peptides
    df['Extracted_Peptides'] = ''
    
    # Process in batches to avoid timeouts
    total_rows = df.shape[0]
    
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch_df = df.iloc[start:end]
        
        for index, row in batch_df.iterrows():
            uniprot_id = row['UniprotID']
            psites = row['pSites']
            
            # Get the full sequence from UniProt
            sequence = get_sequence(uniprot_id)
            
            if sequence:
                # Extract flanking peptides for each phosphorylation site
                peptides = process_psites(psites, sequence)
                df.at[index, 'Extracted_Peptides'] = peptides
            else:
                df.at[index, 'Extracted_Peptides'] = 'Sequence not found'

        # Introduce a delay between batches to avoid overwhelming the server
        print(f"Processed batch {start + 1} to {end}. Waiting {delay} seconds before next batch...")
        time.sleep(delay)


# Read the CSV file
df = pd.read_csv('phosphosites.csv')

# Process the dataframe in batches
process_in_batches(df, batch_size=800, delay=2)

# Optional: Save the updated DataFrame to a new CSV file
df.to_csv('extracted_motifs.csv', index=False)



