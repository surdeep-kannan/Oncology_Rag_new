#!/usr/bin/env python3
import os
import subprocess
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
MASTER_CHUNKS = PROJECT_ROOT / "master_chunks.jsonl"
QA_AUDIT_LOG = PROJECT_ROOT / "qa_audit_log.txt"

def get_processed_files():
    if not MASTER_CHUNKS.exists():
        return set()
    
    processed = set()
    try:
        with open(MASTER_CHUNKS, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if "source_file" in data:
                    processed.add(data["source_file"])
    except Exception as e:
        logger.error(f"Error reading master_chunks.jsonl: {e}")
    return processed

def run_command(cmd, description):
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
        return False
    return True

def process_file(pdf_path):
    pdf_name = pdf_path.name
    logger.info(f"\n{'='*60}\nPROCESSING: {pdf_name}\n{'='*60}")
    
    # Create a dedicated output subdirectory for this PDF to avoid file collisions
    file_out_dir = OUTPUT_DIR / pdf_name.replace(" ", "_")
    file_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Parse
    parsed_out = file_out_dir / "parsed.jsonl"
    if not run_command([
        "python", "scripts/parse_pdfs.py", 
        "--input", str(pdf_path), 
        "--output", str(parsed_out)
    ], "Parsing PDF"):
        return False

    # 2. Build Chunks
    chunks_out = file_out_dir / "chunks.jsonl"
    if not run_command([
        "python", "scripts/build_chunks.py", 
        "--input", str(parsed_out), 
        "--output", str(chunks_out)
    ], "Building Chunks"):
        return False

    # 3. Repair Chunks
    # repair_chunks.py [input] --output-dir [dir]
    if not run_command([
        "python", "scripts/repair_chunks.py", 
        str(chunks_out),
        "--output-dir", str(file_out_dir)
    ], "Repairing Chunks"):
        return False
    repaired_out = file_out_dir / "repaired_chunks.jsonl"

    # 4. Validate Chunks
    # validate_chunks.py [input] --output-dir [dir]
    if not run_command([
        "python", "scripts/validate_chunks.py", 
        str(repaired_out),
        "--output-dir", str(file_out_dir)
    ], "Validating Chunks"):
        return False

    # 5. Append to Master
    logger.info(f"Appending {pdf_name} to master_chunks.jsonl")
    try:
        # We append the repaired chunks which now have validation scores
        with open(repaired_out, "r", encoding="utf-8") as fin:
            with open(MASTER_CHUNKS, "a", encoding="utf-8") as fout:
                fout.write(fin.read())
    except Exception as e:
        logger.error(f"Failed to append to master: {e}")
        return False

    # 6. Audit Search
    logger.info(f"Running audit search for {pdf_name}")
    try:
        # Run search to verify
        search_cmd = [
            "python", "scripts/search_chunks.py",
            "--query", f"What are the key clinical findings or guidelines discussed in {pdf_name}?",
            "--top-k", "1"
        ]
        result = subprocess.run(search_cmd, capture_output=True, text=True)
        
        with open(QA_AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"DOCUMENT: {pdf_name}\n")
            f.write(f"QUERY: What are the key clinical findings or guidelines discussed in this document?\n")
            f.write(f"RESULT:\n{result.stdout}\n")
            f.write(f"{'='*60}\n")
    except Exception as e:
        logger.error(f"Audit search failed: {e}")

    logger.info(f"SUCCESSfully processed {pdf_name}")
    return True

def main():
    processed = get_processed_files()
    all_pdfs = sorted(list(INPUT_DIR.glob("*.pdf")))
    
    # Process all files
    skip_list = []
    
    to_process = [p for p in all_pdfs if p.name not in processed and p.name not in skip_list]
    
    if not to_process:
        logger.info("No new PDFs to process.")
        return

    logger.info(f"Found {len(to_process)} new PDF(s) to process.")
    for pdf in to_process:
        success = process_file(pdf)
        if not success:
            logger.error(f"Stopped batch processing due to failure in {pdf.name}")
            break

if __name__ == "__main__":
    main()
