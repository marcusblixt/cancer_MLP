import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import MLP

EXPRESSION_PATH = "Expression_(Short-read)_Public_26Q1_subsetted.csv"
MUTATION_PATH = "Damaging_Mutations_(Public_26Q1)_subsetted.csv"


def load_dataset(n_input_genes, n_target_genes):
    expression_data = pd.read_csv(EXPRESSION_PATH, index_col=0)
    mutation_data = pd.read_csv(MUTATION_PATH, index_col=0)

    common_index = expression_data.index.intersection(mutation_data.index)
    expression_data = expression_data.loc[common_index]
    mutation_data = mutation_data.loc[common_index]

    # Input features: most-variable expression genes (same pattern as train.py).
    expression_data = expression_data.loc[:, expression_data.var(axis=0) > 0]
    if n_input_genes:
        top_input = expression_data.var(axis=0).sort_values(ascending=False).head(n_input_genes).index
        expression_data = expression_data[top_input]

    # Target genes: the most frequently mutated ones. Predicting genes that are
    # mutated in only a handful of samples is both statistically unlearnable
    # and would dominate the loss with near-constant, uninformative targets.
    mutation_rate = (mutation_data > 0).mean(axis=0)
    target_genes = mutation_rate.sort_values(ascending=False).head(n_target_genes).index
    mutation_data = mutation_data[target_genes]

    X = expression_data.values.astype(np.float32)
    # Binarize: a value of 1 or 2 both mean "damaging mutation present" - this
    # is framed as multi-label binary classification (mutated per gene, not
    # mono- vs biallelic), which is the standard way to pose this problem.
    Y = (mutation_data.values > 0).astype(np.float32)
    return X, Y, list(target_genes)


def main(
    n_input_genes=5000,
    n_target_genes=100,
    hidden_dims=(512, 256),
    dropout=0.3,
    val_split=0.2,
    batch_size=64,
    lr=1e-3,
    epochs=50,
    seed=0,
    device=None,
    output_dir="runs",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device == "cuda" else ""))

    run_dir = Path(output_dir) / f"predict_mutations_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    X, Y, target_genes = load_dataset(n_input_genes, n_target_genes)
    print(f"{X.shape[0]} samples, {X.shape[1]} input genes, {Y.shape[1]} target genes")

    config = dict(
        n_input_genes=n_input_genes, n_target_genes=n_target_genes, hidden_dims=list(hidden_dims),
        dropout=dropout, val_split=val_split, batch_size=batch_size, lr=lr, epochs=epochs, seed=seed,
        n_samples=X.shape[0], n_input_features=X.shape[1], n_target_genes_actual=Y.shape[1],
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "target_genes.json").write_text(json.dumps(target_genes, indent=2))

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_split, random_state=seed)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # The whole dataset is only a few MB (a few thousand samples x a few
    # thousand features) - load it onto the device once so batching doesn't
    # pay a host-to-device transfer cost every step.
    X_train_t = torch.from_numpy(X_train).float().to(device)
    Y_train_t = torch.from_numpy(Y_train).float().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    Y_val_t = torch.from_numpy(Y_val).float().to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, Y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # avoid a possible final batch of size 1, which BatchNorm1d can't train on
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, Y_val_t),
        batch_size=batch_size,
    )

    model = MLP(
        input_dim=X.shape[1],
        hidden_dims=list(hidden_dims),
        num_classes=Y.shape[1],  # one sigmoid output per target gene, not a softmax over classes
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Each target gene is mutated in only a small fraction of samples - plain
    # BCE would be dominated by the "not mutated" negatives and the model
    # would just learn to predict all-zero. pos_weight (computed per gene from
    # the training split only) rebalances the loss toward getting positives right.
    pos_count = Y_train.sum(axis=0)
    neg_count = Y_train.shape[0] - pos_count
    pos_weight = torch.tensor(np.clip(neg_count / np.clip(pos_count, 1, None), 1.0, 50.0), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_logits, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                all_logits.append(logits.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        all_logits, all_targets = np.concatenate(all_logits), np.concatenate(all_targets)

        # Macro-averaged ROC-AUC across target genes - meaningful under severe
        # imbalance, unlike plain accuracy (which "all-zero" would ace).
        val_auroc = _macro_auroc(all_targets, all_logits)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auroc": val_auroc})

        print(
            f"epoch {epoch:3d}/{epochs}  train_loss {train_loss:.4f}  "
            f"val_loss {val_loss:.4f}  val_auroc {val_auroc:.3f}"
        )

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    torch.save(model.state_dict(), run_dir / "model.pt")
    np.savez(run_dir / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

    # Per-gene validation performance: AUROC and average precision, since a
    # single aggregate number hides which genes are actually predictable.
    per_gene = pd.DataFrame({
        "gene": target_genes,
        "mutation_rate": all_targets.mean(axis=0),
        "auroc": [_safe_auroc(all_targets[:, j], all_logits[:, j]) for j in range(all_targets.shape[1])],
        "average_precision": [
            average_precision_score(all_targets[:, j], all_logits[:, j]) if all_targets[:, j].sum() > 0 else np.nan
            for j in range(all_targets.shape[1])
        ],
    }).sort_values("auroc", ascending=False)
    per_gene.to_csv(run_dir / "val_per_gene_metrics.csv", index=False)

    print(f"Saved run artifacts to {run_dir}")
    return model, scaler, target_genes


def _safe_auroc(y_true, y_score):
    # ROC-AUC is undefined for a gene with no positives (or no negatives) in
    # the validation split - happens for rare genes with a small val set.
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def _macro_auroc(y_true, y_score):
    scores = [_safe_auroc(y_true[:, j], y_score[:, j]) for j in range(y_true.shape[1])]
    return float(np.nanmean(scores))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an MLP to predict per-gene damaging-mutation status from gene expression."
    )
    parser.add_argument("--n-input-genes", type=int, default=5000, help="0 to use all expression genes")
    parser.add_argument("--n-target-genes", type=int, default=100, help="most frequently mutated genes to predict")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu - default auto-detects")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        n_input_genes=args.n_input_genes,
        n_target_genes=args.n_target_genes,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        val_split=args.val_split,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
    )
