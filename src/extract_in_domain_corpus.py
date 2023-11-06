import os
import logging
import argparse
import csv
from tqdm import tqdm
import sys

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Filter medical corpus from bigger corpus.")
    parser.add_argument("--in_dir", help="Directory where the input file is located", required=True)
    parser.add_argument("--input_filename", help="Input filename", default="shuffled-pl-en")
    parser.add_argument("--out_dir", help="Directory where the output file will be saved", required=True)
    parser.add_argument("--output_filename", help="Output filename", default="medical_corpus.tsv")
    return parser.parse_args()

def escape_special_characters(text):
    """Escapes tabs and newlines in text fields."""
    return text.replace("\t", "\\t").replace("\n", "\\n")

def main(args):
    input_path = os.path.join(args.in_dir, args.input_filename)
    output_path = os.path.join(args.out_dir, args.output_filename)
    
    _logger.info(f"Processing file: {input_path}")
    _logger.info(f"Writing filtered corpus to file: {output_path}")

    with open(input_path, encoding='utf-8') as input_file, \
         open(output_path, 'w', encoding='utf-8', newline='') as output_file:

        writer = csv.writer(output_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)

        # Write the header
        writer.writerow(["id", "pol", "eng", "src"])

        row_id = 0  # Initialize at 0; it will be incremented to 1 on the first iteration (better to start with 1 for CLI tools like sed)
        issues = []

        for line in tqdm(input_file, desc="Filtering and writing"):
            row_id += 1 
            row = line.rstrip().split('\t')  # Manually split the line by tabs -> necessary or many lines with issues

            if len(row) == 4:
                pol_text, eng_text, corpus_type, src = map(escape_special_characters, row)
                
                if corpus_type == "medical_corpus":
                    # Write the record with the current row_id as id
                    writer.writerow([row_id, pol_text, eng_text, src])
            else:
                # Store issue for logging and continue processing
                issues.append(row_id)

    # Log the issues after processing is complete
    _logger.info("Finished writing medical corpus.")
    if issues:
        _logger.warning(f"Issues found at rows: {issues}")
    else:
        _logger.info("No issues found.")

if __name__ == '__main__':
    args = parse_args()
    main(args)