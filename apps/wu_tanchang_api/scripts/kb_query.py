#!/usr/bin/env python3
"""CLI query script to verify Wu Tanchang KB semantic search."""

import argparse
import sys
from pathlib import Path

# Ensure we can import from apps.wu_tanchang_api and libs/deepagents
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.wu_tanchang_api.agent_factory.kb_search import semantic_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Wu Tanchang KB using semantic search.")
    parser.add_argument("query", type=str, help="Semantic search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--series", type=str, default=None, help="Filter by series (e.g. 商业探店笔记)")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated categories to filter")
    args = parser.parse_args()
    
    series_list = [args.series] if args.series else None
    cats_list = [c.strip() for c in args.categories.split(",")] if args.categories else None
    
    print(f"Querying: '{args.query}' (k={args.k}, series={series_list}, categories={cats_list})...")
    
    try:
        hits = semantic_search(
            query=args.query,
            k=args.k,
            series=series_list,
            categories=cats_list,
            vec_type="note"
        )
        
        if not hits:
            print("No matching notes found.")
            return
            
        print("\n--- Search Results ---")
        for idx, hit in enumerate(hits, 1):
            print(f"\n{idx}. {hit.title} (ID: {hit.note_id})")
            print(f"   Brand:      {hit.brand or 'N/A'}")
            print(f"   Score:      {hit.score:.4f}")
            print(f"   Keywords:   {', '.join(hit.matched_keywords)}")
            print("   Insights:")
            for ins in hit.matched_insights:
                print(f"     - {ins}")
            print(f"   Path:       {hit.raw_path}")
            
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please build the SQLite database and ChromaDB vectors first using:")
        print("  python -m apps.wu_tanchang_api.scripts.kb_build_db")
        print("  python -m apps.wu_tanchang_api.scripts.kb_build_vectors")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
