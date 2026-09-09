"""Hyperparameter search for predict_gene_expression.py.pge

Fixes n_mutation_genes/n_target_genes at the "standard" values, then tunes
hidden_dims and dropout first (the primary levers), and lr/weight_decay
second - a coordinate search, not a full grid over all four, since the
priority between them isn't equal. Each trial is a normal
predict_gene_expression.main() run (same artifacts, same early stopping);
this script just orchestrates the sweep, using a shorter epoch budget while
comparing configs, then retrains the overall winner for longer.
"""
import itertools
import json
from pathlib import Path

import pandas as pd

import predict_gene_expression as pge

N_MUTATION_GENES = 10000
N_TARGET_GENES = 4000

TUNING_DIR = "runs/tuning"
SEARCH_EPOCHS = 100  # shorter budget while comparing configs
SEARCH_PATIENCE = 10
FINAL_EPOCHS = 200  # longer budget for the single winning config
FINAL_PATIENCE = 20

DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4

INPUT_SOURCE = "both"  # mutation + subtype
MUTATION_SELECTION = "univariate"  # frequency or univariate
MIN_MUTATION_COUNT = 10  # univariate selection only: drop genes mutated in fewer than this many training samples

# Stage 1 (primary): architecture and regularization strength.
HIDDEN_DIMS_GRID = [(512, 256, 128), (1024, 512, 256, 128), (2048, 1024, 512, 256, 128)]
DROPOUT_GRID = [0.0, 0.1, 0.2]

# Stage 2 (secondary): optimizer settings, searched around stage 1's winning
# architecture rather than crossed with the full stage 1 grid.
LR_GRID = [1e-3, 5e-4, 1e-4]
WEIGHT_DECAY_GRID = [1e-4, 1e-5]



def run_trial(hidden_dims, dropout, lr, weight_decay, epochs, patience, output_dir, stage, seed=0):
    _, _, _, _, run_dir = pge.main(
        n_mutation_genes=N_MUTATION_GENES,
        n_target_genes=N_TARGET_GENES,
        input_source=INPUT_SOURCE,
        mutation_selection=MUTATION_SELECTION,
        min_mutation_count=MIN_MUTATION_COUNT,
        hidden_dims=tuple(hidden_dims),
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
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
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "best_epoch": int(config["best_epoch"]),
        "val_loss": float(best_row["val_loss"]),
        "val_r2": float(best_row["val_r2"]),
    }


def main():
    Path(TUNING_DIR).mkdir(parents=True, exist_ok=True)
    results = []

    # --- Stage 1: hidden_dims x dropout, lr/weight_decay held at defaults ---
    stage1_grid = list(itertools.product(HIDDEN_DIMS_GRID, DROPOUT_GRID))
    print(
        f"=== Stage 1: {len(stage1_grid)} (hidden_dims, dropout) combos "
        f"(lr={DEFAULT_LR}, weight_decay={DEFAULT_WEIGHT_DECAY} fixed) ==="
    )
    for i, (hidden_dims, dropout) in enumerate(stage1_grid, 1):
        print(f"\n[stage 1: {i}/{len(stage1_grid)}] hidden_dims={hidden_dims} dropout={dropout}")
        result = run_trial(
            hidden_dims, dropout, DEFAULT_LR, DEFAULT_WEIGHT_DECAY, SEARCH_EPOCHS, SEARCH_PATIENCE, TUNING_DIR, stage=1
        )
        results.append(result)
        print(f"  -> val_loss={result['val_loss']:.4f}  val_r2={result['val_r2']:.4f}")

    stage1_best = min(results, key=lambda r: r["val_loss"])
    print(
        f"\nStage 1 winner: hidden_dims={stage1_best['hidden_dims']} dropout={stage1_best['dropout']} "
        f"(val_loss={stage1_best['val_loss']:.4f})"
    )

    # --- Stage 2: lr x weight_decay, hidden_dims/dropout fixed to stage 1's winner ---
    stage2_grid = [
        (lr, wd)
        for lr, wd in itertools.product(LR_GRID, WEIGHT_DECAY_GRID)
        if not (lr == DEFAULT_LR and wd == DEFAULT_WEIGHT_DECAY)  # already covered by stage 1's winning trial
    ]
    print(
        f"\n=== Stage 2: {len(stage2_grid)} (lr, weight_decay) combos "
        f"(hidden_dims={stage1_best['hidden_dims']}, dropout={stage1_best['dropout']} fixed) ==="
    )
    for i, (lr, weight_decay) in enumerate(stage2_grid, 1):
        print(f"\n[stage 2: {i}/{len(stage2_grid)}] lr={lr} weight_decay={weight_decay}")
        result = run_trial(
            stage1_best["hidden_dims"], stage1_best["dropout"], lr, weight_decay,
            SEARCH_EPOCHS, SEARCH_PATIENCE, TUNING_DIR, stage=2,
        )
        results.append(result)
        print(f"  -> val_loss={result['val_loss']:.4f}  val_r2={result['val_r2']:.4f}")

    summary = pd.DataFrame(results).sort_values("val_loss").reset_index(drop=True)
    summary.to_csv(Path(TUNING_DIR) / "tuning_summary.csv", index=False)
    print("\n=== Leaderboard (best val_loss first) ===")
    print(summary.to_string(index=False))

    best = summary.iloc[0]
    print(
        f"\nBest overall: hidden_dims={best['hidden_dims']} dropout={best['dropout']} "
        f"lr={best['lr']} weight_decay={best['weight_decay']} (val_loss={best['val_loss']:.4f})"
    )

    # Retrain the winning config with a longer budget, saved to the normal
    # runs/ folder (not runs/tuning) as a proper standalone run.
    print(f"\nRetraining the winning config with a longer budget ({FINAL_EPOCHS} epochs, patience={FINAL_PATIENCE}) -> runs/ ...")
    final = run_trial(
        best["hidden_dims"], best["dropout"], best["lr"], best["weight_decay"],
        FINAL_EPOCHS, FINAL_PATIENCE, "runs", stage="final",
    )
    print(f"Final run: {final['run_dir']}  val_loss={final['val_loss']:.4f}  val_r2={final['val_r2']:.4f}")


if __name__ == "__main__":
    main()
