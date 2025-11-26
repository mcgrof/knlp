#!/usr/bin/env python3
"""
Query best perplexity results from gpt2-mla-compressed-latents W&B project.
"""

import wandb
import sys

def get_all_runs(entity, project):
    """Fetch all runs from project and extract best perplexity."""
    api = wandb.Api()

    print(f"Fetching runs from {entity}/{project}...")
    runs = api.runs(f"{entity}/{project}")

    results = []

    for run in runs:
        run_name = run.name
        display_name = run.config.get('display_name', run_name)
        state = run.state

        # Try multiple field names for perplexity
        best_val_ppl = (
            run.summary.get('final/best_val_perplexity') or
            run.summary.get('val/perplexity') or
            run.summary.get('best_val_perplexity')
        )

        steps = run.summary.get('_step', 'N/A')

        results.append({
            'display_name': display_name,
            'run_name': run_name,
            'state': state,
            'best_val_perplexity': best_val_ppl,
            'steps': steps,
            'url': run.url
        })

    return results


def main():
    entity = "mcgrof-citizen"
    project = "gpt2-mla-compressed-latents"

    results = get_all_runs(entity, project)

    # Filter out runs without perplexity
    valid_results = [r for r in results if r['best_val_perplexity'] is not None]

    # Sort by perplexity (lower is better)
    valid_results.sort(key=lambda x: x['best_val_perplexity'])

    print("\n" + "="*80)
    print(f"Best Perplexity Results from {project}")
    print("="*80)
    print(f"\nTotal runs: {len(results)}")
    print(f"Runs with perplexity: {len(valid_results)}")

    print("\n" + "-"*80)
    print(f"{'Rank':<6} {'Display Name':<40} {'Perplexity':<12} {'Steps':<8} {'State':<10}")
    print("-"*80)

    for i, result in enumerate(valid_results, 1):
        print(f"{i:<6} {result['display_name']:<40} {result['best_val_perplexity']:<12.2f} "
              f"{result['steps']:<8} {result['state']:<10}")

    print("\n" + "="*80)
    print("Best Model:")
    print("="*80)
    best = valid_results[0]
    print(f"  Display Name: {best['display_name']}")
    print(f"  Run Name: {best['run_name']}")
    print(f"  Perplexity: {best['best_val_perplexity']:.2f}")
    print(f"  Steps: {best['steps']}")
    print(f"  URL: {best['url']}")

    # Check if MLAKV2 is in the results
    print("\n" + "="*80)
    print("MLAKV2 Variants:")
    print("="*80)
    mlakv2_runs = [r for r in valid_results if 'MLAKV2' in r['display_name'] or 'mlakv2' in r['run_name'].lower()]

    if mlakv2_runs:
        for run in mlakv2_runs:
            print(f"  {run['display_name']:<40} Perplexity: {run['best_val_perplexity']:.2f}")
    else:
        print("  No MLAKV2 runs found in this project")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
