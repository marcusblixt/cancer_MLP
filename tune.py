"""Hyperparameter search for predict_gene_expression.py.

ARCHITECTURE/PRETRAIN_SUBTYPE (and the other non-grid settings near the top)
are fixed for the whole sweep, not tuned - set them once for the run you
want. The tunable parameters are searched as a sequence of small coordinate
searches rather than one big grid, since their priority isn't equal:

  Stage 1 (primary): shared-trunk hidden_dims x dropout - the architecture's
    overall capacity and regularization strength.
  Stage 2 (primary, two_head only): hidden_dims_mutation x hidden_dims_subtype
    - how much of that capacity each branch gets before merging, searched
    around stage 1's winning shared trunk.
  Stage 3 (primary): min_mutation_count x l1_mutation - how strict the
    candidate-gene floor is and how hard the model is pushed to prune
    unhelpful mutation genes on its own, searched around stage 1+2's winners.
  Stage 4 (secondary): lr x weight_decay - optimizer settings, searched
    around the winning architecture rather than crossed with the earlier
    grids.

Each trial is a normal predict_gene_expression.main() run (same artifacts,
same early stopping); this script just orchestrates the sweep, using a
shorter epoch budget while comparing configs, then retrains the overall
winner (best val_loss across every trial from every stage) for longer.
"""
import itertools
import json
from pathlib import Path

import pandas as pd

import train as pge

N_MUTATION_GENES = 0
N_TARGET_GENES = 0

RUN_DIR = "all_targets"
SEARCH_EPOCHS = 100  # shorter budget while comparing configs
SEARCH_PATIENCE = 10
FINAL_EPOCHS = 200  # longer budget for the single winning config
FINAL_PATIENCE = 20

DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4

INPUT_SOURCE = "both"  # mutation + subtype
MUTATION_SELECTION = "frequency"  # frequency or univariate

ARCHITECTURE = "two_head"  # "single" or "two_head"
PRETRAIN_SUBTYPE = True  # two_head only - pretrain subtype branch + shared trunk before adding the mutation branch
PRETRAIN_EPOCHS = 100
PRETRAIN_PATIENCE = 10

# Defaults used for every tunable parameter until its own stage overrides it.
DEFAULT_MIN_MUTATION_COUNT = 10
DEFAULT_HIDDEN_DIMS_MUTATION = (512, 256)
DEFAULT_HIDDEN_DIMS_SUBTYPE = (128,)
DEFAULT_L1_MUTATION = 1e-4

# Stage 1 (primary): shared-trunk architecture and regularization strength.
HIDDEN_DIMS_GRID = [(256, 128), (512, 256), (512, 256, 128)]
DROPOUT_GRID = [0.0, 0.1, 0.3]

# Stage 2 (primary, two_head only): per-branch capacity.
HIDDEN_DIMS_MUTATION_GRID = [(256, 128), (512, 256), (1024, 512, 256)]
HIDDEN_DIMS_SUBTYPE_GRID = [(64,), (128,), (256, 128)]

# Stage 3 (primary): mutation-gene selection strictness and pruning pressure.
MIN_MUTATION_COUNT_GRID = [4, 6, 8]
L1_MUTATION_GRID = [0.0, 1e-4, 1e-3]  # l1_mutation only applies to architecture="two_head"

# Stage 4 (secondary): optimizer settings.
LR_GRID = [1e-3, 5e-4]
WEIGHT_DECAY_GRID = [1e-4, 1e-5]


def run_trial(
    hidden_dims, dropout, lr, weight_decay, epochs, patience, output_dir, stage,
    hidden_dims_mutation=DEFAULT_HIDDEN_DIMS_MUTATION, hidden_dims_subtype=DEFAULT_HIDDEN_DIMS_SUBTYPE,
    min_mutation_count=DEFAULT_MIN_MUTATION_COUNT, l1_mutation=DEFAULT_L1_MUTATION, seed=0,
):
    _, _, _, _, run_dir = pge.main(
        n_mutation_genes=N_MUTATION_GENES,
        n_target_genes=N_TARGET_GENES,
        input_source=INPUT_SOURCE,
        mutation_selection=MUTATION_SELECTION,
        min_mutation_count=min_mutation_count,
        architecture=ARCHITECTURE,
        hidden_dims=tuple(hidden_dims),
        hidden_dims_mutation=tuple(hidden_dims_mutation),
        hidden_dims_subtype=tuple(hidden_dims_subtype),
        dropout=dropout,
        weight_decay=weight_decay,
        pretrain_subtype=PRETRAIN_SUBTYPE if ARCHITECTURE == "two_head" else False,
        pretrain_epochs=PRETRAIN_EPOCHS,
        pretrain_patience=PRETRAIN_PATIENCE,
        l1_mutation=l1_mutation if ARCHITECTURE == "two_head" else 0.0,
        lr=lr,
        epochs=epochs,
        patience=patience,
        seed=seed,
        output_dir=output_dir,
    )
    config = json.loads((run_dir / "config.json").read_text())
    history = pd.read_csv(run_dir / "history.csv")
    best_row = history.loc[history["epoch"] == config["best_epoch"]].iloc[0]
    return {
        "stage": stage,
        "run_dir": run_dir.name,
        "hidden_dims": tuple(hidden_dims),
        "hidden_dims_mutation": tuple(hidden_dims_mutation),
        "hidden_dims_subtype": tuple(hidden_dims_subtype),
        "dropout": dropout,
        "min_mutation_count": min_mutation_count,
        "l1_mutation": l1_mutation,
        "lr": lr,
        "weight_decay": weight_decay,
        "best_epoch": int(config["best_epoch"]),
        "val_loss": float(best_row["val_loss"]),
        "val_r2": float(best_row["val_r2"]),
    }


def main():
    tuning_dir = RUN_DIR + "/tuning"
    results = []

    # --- Stage 1: hidden_dims (shared trunk) x dropout ---
    stage1_grid = list(itertools.product(HIDDEN_DIMS_GRID, DROPOUT_GRID))
    print(f"=== Stage 1: {len(stage1_grid)} (hidden_dims, dropout) combos ===")
    for i, (hidden_dims, dropout) in enumerate(stage1_grid, 1):
        print(f"\n[stage 1: {i}/{len(stage1_grid)}] hidden_dims={hidden_dims} dropout={dropout}")
        result = run_trial(hidden_dims, dropout, DEFAULT_LR, DEFAULT_WEIGHT_DECAY, SEARCH_EPOCHS, SEARCH_PATIENCE, tuning_dir, stage=1)
        results.append(result)
        print(f"  -> val_loss={result['val_loss']:.4f}  val_r2={result['val_r2']:.4f}")

    stage1_best = min((r for r in results if r["stage"] == 1), key=lambda r: r["val_loss"])
    print(f"\nStage 1 winner: hidden_dims={stage1_best['hidden_dims']} dropout={stage1_best['dropout']} (val_loss={stage1_best['val_loss']:.4f})")

    # --- Stage 2 (two_head only): hidden_dims_mutation x hidden_dims_subtype ---
    if ARCHITECTURE == "two_head":
        stage2_grid = [
            (hd_mut, hd_sub)
            for hd_mut, hd_sub in itertools.product(HIDDEN_DIMS_MUTATION_GRID, HIDDEN_DIMS_SUBTYPE_GRID)
            if not (hd_mut == DEFAULT_HIDDEN_DIMS_MUTATION and hd_sub == DEFAULT_HIDDEN_DIMS_SUBTYPE)  # already covered by stage 1
        ]
        print(f"\n=== Stage 2: {len(stage2_grid)} (hidden_dims_mutation, hidden_dims_subtype) combos (shared trunk fixed to stage 1 winner) ===")
        for i, (hd_mut, hd_sub) in enumerate(stage2_grid, 1):
            print(f"\n[stage 2: {i}/{len(stage2_grid)}] hidden_dims_mutation={hd_mut} hidden_dims_subtype={hd_sub}")
            result = run_trial(
                stage1_best["hidden_dims"], stage1_best["dropout"], DEFAULT_LR, DEFAULT_WEIGHT_DECAY,
                SEARCH_EPOCHS, SEARCH_PATIENCE, tuning_dir, stage=2,
                hidden_dims_mutation=hd_mut, hidden_dims_subtype=hd_sub,
            )
            results.append(result)
            print(f"  -> val_loss={result['val_loss']:.4f}  val_r2={result['val_r2']:.4f}")
        stage2_best = min((r for r in results if r["stage"] in (1, 2)), key=lambda r: r["val_loss"])
    else:
        print("\n=== Stage 2 skipped: hidden_dims_mutation/hidden_dims_subtype only apply to architecture='two_head' ===")
        stage2_best = stage1_best
    print(f"\nStage 2 winner: hidden_dims_mutation={stage2_best['hidden_dims_mutation']} hidden_dims_subtype={stage2_best['hidden_dims_subtype']} (val_loss={stage2_best['val_loss']:.4f})")

    # --- Stage 3: min_mutation_count x l1_mutation ---
    l1_grid = L1_MUTATION_GRID if ARCHITECTURE == "two_head" else [0.0]
    stage3_grid = [
        (mmc, l1)
        for mmc, l1 in itertools.product(MIN_MUTATION_COUNT_GRID, l1_grid)
        if not (mmc == DEFAULT_MIN_MUTATION_COUNT and l1 == DEFAULT_L1_MUTATION)  # already covered by stage 2
    ]
    print(f"\n=== Stage 3: {len(stage3_grid)} (min_mutation_count, l1_mutation) combos ===")
    for i, (mmc, l1) in enumerate(stage3_grid, 1):
        print(f"\n[stage 3: {i}/{len(stage3_grid)}] min_mutation_count={mmc} l1_mutation={l1}")
        result = run_trial(
            stage1_best["hidden_dims"], stage1_best["dropout"], DEFAULT_LR, DEFAULT_WEIGHT_DECAY,
            SEARCH_EPOCHS, SEARCH_PATIENCE, tuning_dir, stage=3,
            hidden_dims_mutation=stage2_best["hidden_dims_mutation"], hidden_dims_subtype=stage2_best["hidden_dims_subtype"],
            min_mutation_count=mmc, l1_mutation=l1,
        )
        results.append(result)
        print(f"  -> val_loss={result['val_loss']:.4f}  val_r2={result['val_r2']:.4f}")
    stage3_best = min((r for r in results if r["stage"] in (2, 3)), key=lambda r: r["val_loss"])
    print(f"\nStage 3 winner: min_mutation_count={stage3_best['min_mutation_count']} l1_mutation={stage3_best['l1_mutation']} (val_loss={stage3_best['val_loss']:.4f})")

    # --- Stage 4: lr x weight_decay, everything else fixed to stage 1-3's winners ---
    stage4_grid = [
        (lr, wd)
        for lr, wd in itertools.product(LR_GRID, WEIGHT_DECAY_GRID)
        if not (lr == DEFAULT_LR and wd == DEFAULT_WEIGHT_DECAY)  # already covered by stage 3
    ]
    print(f"\n=== Stage 4: {len(stage4_grid)} (lr, weight_decay) combos ===")
    for i, (lr, weight_decay) in enumerate(stage4_grid, 1):
        print(f"\n[stage 4: {i}/{len(stage4_grid)}] lr={lr} weight_decay={weight_decay}")
        result = run_trial(
            stage1_best["hidden_dims"], stage1_best["dropout"], lr, weight_decay,
            SEARCH_EPOCHS, SEARCH_PATIENCE, tuning_dir, stage=4,
            hidden_dims_mutation=stage2_best["hidden_dims_mutation"], hidden_dims_subtype=stage2_best["hidden_dims_subtype"],
            min_mutation_count=stage3_best["min_mutation_count"], l1_mutation=stage3_best["l1_mutation"],
        )
        results.append(result)
        print(f"  -> val_loss={result['val_loss']:.4f}  val_r2={result['val_r2']:.4f}")

    summary = pd.DataFrame(results).sort_values("val_loss").reset_index(drop=True)
    summary.to_csv(Path("runs") / tuning_dir / "tuning_summary.csv", index=False)
    print("\n=== Leaderboard (best val_loss first, across all stages) ===")
    print(summary.to_string(index=False))

    best = summary.iloc[0]
    print(
        f"\nBest overall: hidden_dims={best['hidden_dims']} hidden_dims_mutation={best['hidden_dims_mutation']} "
        f"hidden_dims_subtype={best['hidden_dims_subtype']} dropout={best['dropout']} "
        f"min_mutation_count={best['min_mutation_count']} l1_mutation={best['l1_mutation']} "
        f"lr={best['lr']} weight_decay={best['weight_decay']} (val_loss={best['val_loss']:.4f})"
    )

    # Retrain the winning config with a longer budget, saved to the normal
    # RUN_DIR folder (not RUN_DIR/tuning) as a proper standalone run.
    print(f"\nRetraining the winning config with a longer budget ({FINAL_EPOCHS} epochs, patience={FINAL_PATIENCE}) -> {RUN_DIR}/ ...")
    final = run_trial(
        best["hidden_dims"], best["dropout"], best["lr"], best["weight_decay"],
        FINAL_EPOCHS, FINAL_PATIENCE, RUN_DIR, stage="final",
        hidden_dims_mutation=best["hidden_dims_mutation"], hidden_dims_subtype=best["hidden_dims_subtype"],
        min_mutation_count=best["min_mutation_count"], l1_mutation=best["l1_mutation"],
    )
    print(f"Final run: {final['run_dir']}  val_loss={final['val_loss']:.4f}  val_r2={final['val_r2']:.4f}")


if __name__ == "__main__":
    main()
