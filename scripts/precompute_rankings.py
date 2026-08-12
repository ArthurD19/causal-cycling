"""
Batch computation of DML (ATE) + Causal Forest (mean CATE) for all WorldTour riders.
Results are saved incrementally to riders_ate_results.csv.
Re-running the script skips already-computed riders (resumable).

Usage:
    python scripts/precompute_rankings.py
    python scripts/precompute_rankings.py --no-cf      # DML only, faster
    python scripts/precompute_rankings.py --reset      # recompute everything from scratch
"""

import argparse
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import causal_model as cm

OUTPUT_FILE = BASE_DIR / 'data_app' / 'riders_ate_results.csv'
N_BOOT      = 100   # reduced vs default (200) for speed; enough for stable CIs
N_TREES     = 100   # reduced vs default (300) for CF ranking; stable enough for comparison

COLUMNS = [
    'rider', 'equipe',
    'ate_log', 'ate_orig', 'ci_low', 'ci_high', 'significant',
    'r2_t', 'r2_y',
    'mean_cate', 'std_cate', 'p25_cate', 'p75_cate',
    'n_obs', 'n_selected',
]


def load_existing() -> pd.DataFrame:
    if OUTPUT_FILE.exists():
        return pd.read_csv(OUTPUT_FILE)
    return pd.DataFrame(columns=COLUMNS)


def save_row(row: dict, df_existing: pd.DataFrame) -> pd.DataFrame:
    new_row = pd.DataFrame([row])
    df_out = pd.concat([df_existing, new_row], ignore_index=True)
    df_out.to_csv(OUTPUT_FILE, index=False)
    return df_out


def compute_rider(rider: str, equipe_list: list, run_cf: bool) -> dict | None:
    df = cm.load_rider(rider, equipe=equipe_list)
    if df is None:
        return None
    prep = cm.prepare_features(df)
    if prep is None:
        return None
    X, T, Y, df_clean, feats = prep

    dml = cm.run_dml(X, T, Y, n_boot=N_BOOT)
    if dml is None:
        return None

    row = {
        'rider':       rider,
        'equipe':      equipe_list[0] if equipe_list else '',
        'ate_log':     round(dml['ate_log'], 6),
        'ate_orig':    round(dml['ate_orig'], 4),
        'ci_low':      round(dml['ci_low'], 4),
        'ci_high':     round(dml['ci_high'], 4),
        'significant': int(dml['significant']),
        'r2_t':        round(dml['r2_t'], 4),
        'r2_y':        round(dml['r2_y'], 4),
        'mean_cate':   None,
        'std_cate':    None,
        'p25_cate':    None,
        'p75_cate':    None,
        'n_obs':       int(len(T)),
        'n_selected':  int(T.sum()),
    }

    if run_cf:
        try:
            cf_model, cates = cm.run_causal_forest(X, T, Y, n_trees=N_TREES)
            row['mean_cate'] = round(float(np.mean(cates)), 4)
            row['std_cate']  = round(float(np.std(cates)), 4)
            row['p25_cate']  = round(float(np.percentile(cates, 25)), 4)
            row['p75_cate']  = round(float(np.percentile(cates, 75)), 4)
        except Exception as e:
            print(f"    CF failed: {e}")

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cf', action='store_true', help='Also run Causal Forest (slower)')
    parser.add_argument('--reset', action='store_true', help='Recompute everything from scratch')
    args = parser.parse_args()

    run_cf = args.cf

    if args.reset and OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print("Reset: existing results deleted.")

    df_done = load_existing()
    done_pairs = set(zip(df_done['rider'], df_done['equipe'])) if len(df_done) > 0 else set()
    print(f"Already computed: {len(done_pairs)} rider-team pairs")

    # Build list of all WorldTour (rider, equipe) pairs
    all_pairs = []
    for canonical, variants in cm.TEAM_GROUPS.items():
        all_variants = list(cm.expand_team(canonical))
        riders = cm.find_team_riders(all_variants, min_selections=10)
        for rider in riders:
            all_pairs.append((rider, canonical, all_variants))

    # Deduplicate riders across teams (keep canonical team)
    seen_riders = set()
    pairs_deduped = []
    for rider, canonical, variants in all_pairs:
        if rider not in seen_riders:
            pairs_deduped.append((rider, canonical, variants))
            seen_riders.add(rider)

    total = len(pairs_deduped)
    todo = [(r, c, v) for r, c, v in pairs_deduped if (r, c) not in done_pairs]
    cf_info = f"yes (n_trees={N_TREES})" if run_cf else "no"
    print(f"Total WorldTour riders: {total}  |  To compute: {len(todo)}")
    print(f"CF: {cf_info}  |  n_boot={N_BOOT}")
    print()

    n_already_done = len(done_pairs)
    t0_global = time.time()
    for i, (rider, canonical, variants) in enumerate(todo):
        t0 = time.time()
        current = n_already_done + i + 1
        pct = current / total * 100
        print(f"[{current}/{total}  {pct:.0f}%]  {rider} ({canonical}) ...", end=' ', flush=True)

        try:
            row = compute_rider(rider, variants, run_cf=run_cf)
            if row is None:
                print("skipped (not enough data)")
                continue
            row['equipe'] = canonical
            df_done = save_row(row, df_done)
            done_pairs.add((rider, canonical))
            elapsed = time.time() - t0
            sig = "✓" if row['significant'] else "✗"
            print(f"ATE={row['ate_orig']:+.3f}  sig={sig}  ({elapsed:.1f}s)")
        except KeyboardInterrupt:
            print("\n\nInterrupted — progress saved.")
            sys.exit(0)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # ETA
        if (i + 1) % 10 == 0:
            elapsed_total = time.time() - t0_global
            rate = (i + 1) / elapsed_total
            eta_sec = (len(todo) - i - 1) / rate
            eta_min = eta_sec / 60
            print(f"  → ETA: {eta_min:.0f} min remaining")

    total_time = (time.time() - t0_global) / 60
    print(f"\nDone! {len(df_done)} riders computed in {total_time:.1f} min.")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
