#!/usr/bin/env python3
"""inspect_chunks.py – Inspect and evaluate chunk quality.

Usage:
    python scripts/inspect_chunks.py
    python scripts/inspect_chunks.py --stats
    python scripts/inspect_chunks.py --sample 5
    python scripts/inspect_chunks.py --heading "Treatment"
    python scripts/inspect_chunks.py --eval
    python scripts/inspect_chunks.py --tree
"""
from __future__ import annotations
import argparse, json, logging, random, re, sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from medrag import config as cfg
from medrag.logging_setup import setup_logging
from medrag.utils.visualization import print_chunk_stats

def parse_args():
    p = argparse.ArgumentParser(description="Inspect chunk quality.")
    p.add_argument("--input","-i",type=Path,default=None)
    p.add_argument("--stats",action="store_true")
    p.add_argument("--sample","-n",type=int,default=0)
    p.add_argument("--heading",type=str,default=None)
    p.add_argument("--level1",type=str,default=None)
    p.add_argument("--level2",type=str,default=None)
    p.add_argument("--chunk-id",type=int,default=None)
    p.add_argument("--eval",action="store_true")
    p.add_argument("--tree",action="store_true")
    p.add_argument("--search",type=str,default=None)
    p.add_argument("--max-display",type=int,default=20)
    return p.parse_args()

def load_chunks(path):
    chunks = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): chunks.append(json.loads(line))
    return chunks

def filter_chunks(chunks,heading=None,level1=None,level2=None,chunk_id=None,search=None):
    r = chunks
    if chunk_id is not None: r=[c for c in r if c.get("chunk_id")==chunk_id]
    if heading: hl=heading.lower(); r=[c for c in r if hl in (c.get("heading") or "").lower()]
    if level1: l=level1.lower(); r=[c for c in r if l in (c.get("level1") or "").lower()]
    if level2: l=level2.lower(); r=[c for c in r if l in (c.get("level2") or "").lower()]
    if search: s=search.lower(); r=[c for c in r if s in (c.get("content") or "").lower()]
    return r

def display_chunk(chunk, verbose=True):
    cid=chunk.get("chunk_id","?"); h=chunk.get("heading","")
    wc=chunk.get("token_count",0); content=chunk.get("content","")
    pages = f"p{chunk.get('page_start','?')}"
    if chunk.get("page_end") and chunk["page_end"]!=chunk.get("page_start"):
        pages+=f"-{chunk['page_end']}"
    print(f"\n┌─ Chunk #{cid} {'─'*40}")
    print(f"│ Heading: {h}")
    print(f"│ L1: {chunk.get('level1','—')} | L2: {chunk.get('level2','—')} | L3: {chunk.get('level3','—')}")
    print(f"│ Pages: {pages} | Words: {wc}")
    print(f"├─ Content {'─'*40}")
    if verbose:
        for line in content.split("\n"): print(f"│  {line}")
    else:
        preview=content[:200].replace("\n"," ")
        if len(content)>200: preview+="..."
        print(f"│  {preview}")
    print(f"└{'─'*50}")

def evaluate_quality(chunks):
    print(f"\n{'═'*50}\n  QUALITY EVALUATION\n{'═'*50}")
    issues = defaultdict(list)
    for c in chunks:
        cid=c.get("chunk_id",0); content=c.get("content","")
        wc=c.get("token_count",len(content.split()))
        if wc<40: issues["Too short (<40w)"].append(cid)
        if wc>800: issues["Too long (>800w)"].append(cid)
        if not c.get("heading","").strip(): issues["Empty heading"].append(cid)
        if not c.get("level1"): issues["Missing level1"].append(cid)
        if content and content[0].islower(): issues["Starts lowercase"].append(cid)
        if re.search(r"[a-z][A-Z][a-z]",content): issues["OCR merge"].append(cid)
    total=len(chunks); print(f"\n  Evaluated: {total}\n")
    if not issues: print("  ✅ No issues!")
    else:
        for issue,ids in sorted(issues.items()):
            ex=", ".join(str(i) for i in ids[:5])
            if len(ids)>5: ex+=f" (+{len(ids)-5})"
            print(f"  ⚠  {issue}: {len(ids)} ({len(ids)*100/total:.1f}%) – IDs: {ex}")
    print(f"{'═'*50}")

def show_tree(chunks):
    print(f"\n{'═'*60}\n  HIERARCHY TREE\n{'═'*60}")
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for c in chunks:
        l1=c.get("level1") or "(none)"; l2=c.get("level2") or "(none)"
        l3=c.get("level3") or c.get("heading") or "(none)"
        tree[l1][l2][l3].append(c.get("chunk_id",0))
    for l1,l2d in tree.items():
        print(f"\n  ├── {l1}")
        for l2,l3d in l2d.items():
            if l2!="(none)": print(f"  │   ├── {l2}")
            for l3,cids in l3d.items():
                if l3!="(none)" and l3!=l2:
                    print(f"  │   │   ├── {l3} ({len(cids)} chunks)")
    print(f"{'═'*60}")

def main():
    args=parse_args(); setup_logging(level="WARNING")
    path=args.input or cfg.output_dir()/"chunks.jsonl"
    if not path.exists(): print(f"❌ Not found: {path}"); sys.exit(1)
    chunks=load_chunks(path); print(f"Loaded {len(chunks)} chunks")
    filtered=filter_chunks(chunks,args.heading,args.level1,args.level2,args.chunk_id,args.search)
    if len(filtered)!=len(chunks): print(f"Filtered: {len(filtered)}")
    if args.stats or (not args.sample and not args.eval and not args.tree and not args.heading and not args.level1 and not args.level2 and args.chunk_id is None and not args.search):
        print_chunk_stats(filtered)
    if args.sample>0:
        for c in random.sample(filtered,min(args.sample,len(filtered))): display_chunk(c)
    if args.heading or args.level1 or args.level2 or args.chunk_id is not None or args.search:
        for c in filtered[:args.max_display]: display_chunk(c,verbose=args.chunk_id is not None)
        if len(filtered)>args.max_display: print(f"  ... +{len(filtered)-args.max_display} more")
    if args.eval: evaluate_quality(filtered)
    if args.tree: show_tree(filtered)

if __name__=="__main__": main()
