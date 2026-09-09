import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import MLP

EXPRESSION_PATH = "Expression_(Short-read)_Public_26Q1_subsetted.csv"
MUTATION_PATH = "Damaging_Mutations_(Public_26Q1)_subsetted.csv"
SUBTYPE_PATH = "Subtype_Matrix_Public_26Q1_subsetted.csv"

# Same heuristic as explore.ipynb: exclude subtype-matrix columns that look
# like molecular-alteration flags (gene loss-of-function, mutation
# status) rather than tissue/lineage, then give each sample the largest
# remaining group it belongs to as its representative subtype.
EXCLUDE_SUBTYPES = ["MSI", "KRAS", "NRAS", "HRAS", "PIK3CA"]


def get_sample_subtypes(subtype_path=SUBTYPE_PATH):
    subtype_data = pd.read_csv(subtype_path, index_col=0)
    exclude_cols = subtype_data.columns[
        subtype_data.columns.str.contains(r"_LoF|p\.", regex=True)
    ].tolist() + EXCLUDE_SUBTYPES
    lineage_cols = subtype_data.columns.difference(exclude_cols)
    group_sizes = subtype_data[lineage_cols].sum(axis=0)

    def pick_primary_subtype(row):
        positive = row[row == 1].index.intersection(lineage_cols)
        return group_sizes[positive].idxmax() if len(positive) else "Unknown"

    return subtype_data.apply(pick_primary_subtype, axis=1)


def _load_features(path, n_top_genes):
    data = pd.read_csv(path, index_col=0)
    data = data.loc[:, data.var(axis=0) > 0]
    if n_top_genes:
        top_genes = data.var(axis=0).sort_values(ascending=False).head(n_top_genes).index
        data = data[top_genes]
    return data


def load_dataset(data_source, n_top_genes, min_samples_per_class):
    if data_source == "both":
        expr = _load_features(EXPRESSION_PATH, n_top_genes)
        mut = _load_features(MUTATION_PATH, n_top_genes)
        # Only samples present in both panels can use the combined features.
        common_index = expr.index.intersection(mut.index)
        expr, mut = expr.loc[common_index], mut.loc[common_index]
        # Both panels use gene symbols as column names - suffix to keep them
        # distinct once concatenated (e.g. TP53_expr vs TP53_mut).
        data = pd.concat([expr.add_suffix("_expr"), mut.add_suffix("_mut")], axis=1)
    else:
        path = EXPRESSION_PATH if data_source == "expression" else MUTATION_PATH
        data = _load_features(path, n_top_genes)

    labels = get_sample_subtypes().reindex(data.index)
    class_counts = labels.value_counts()
    large_classes = class_counts[
        (class_counts >= min_samples_per_class) & (class_counts.index != "Unknown")
    ].index
    keep = labels.isin(large_classes)

    X = data.loc[keep].values.astype(np.float32)
    y = labels.loc[keep].values
    return X, y, sorted(large_classes)


def main(
    data_source="expression",
    n_top_genes=5000,
    hidden_dims=(256, 64),
    dropout=0.3,
    min_samples_per_class=20,
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

    run_dir = Path(output_dir) / f"{data_source}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    X, y, classes = load_dataset(data_source, n_top_genes, min_samples_per_class)
    print(f"{data_source}: {X.shape[0]} samples, {X.shape[1]} features, {len(classes)} classes")

    config = dict(
        data_source=data_source, n_top_genes=n_top_genes, hidden_dims=list(hidden_dims),
        dropout=dropout, min_samples_per_class=min_samples_per_class, val_split=val_split,
        batch_size=batch_size, lr=lr, epochs=epochs, seed=seed,
        n_samples=X.shape[0], n_features=X.shape[1], n_classes=len(classes),
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    label_encoder = LabelEncoder().fit(classes)
    y_encoded = label_encoder.transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=val_split, stratify=y_encoded, random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # avoid a possible final batch of size 1, which BatchNorm1d can't train on
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
        batch_size=batch_size,
    )

    model = MLP(
        input_dim=X.shape[1],
        hidden_dims=list(hidden_dims),
        num_classes=len(classes),
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        print(
            f"epoch {epoch:3d}/{epochs}  train_loss {train_loss:.4f}  "
            f"val_loss {val_loss:.4f}  val_acc {val_acc:.3f}"
        )

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "classes.json").write_text(json.dumps(label_encoder.classes_.tolist(), indent=2))
    np.savez(run_dir / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

    # Per-class validation performance, not just the aggregate accuracy above.
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            all_preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
            all_targets.append(yb.numpy())
    all_preds, all_targets = np.concatenate(all_preds), np.concatenate(all_targets)

    target_names = label_encoder.classes_
    report = classification_report(all_targets, all_preds, labels=range(len(target_names)),
                                    target_names=target_names, zero_division=0)
    (run_dir / "val_classification_report.txt").write_text(report)

    cm = confusion_matrix(all_targets, all_preds, labels=range(len(target_names)))
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(run_dir / "val_confusion_matrix.csv")

    print(f"Saved run artifacts to {run_dir}")
    return model, scaler, label_encoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an MLP to predict cancer subtype from expression, mutation, or both combined."
    )
    parser.add_argument("--data-source", choices=["expression", "mutation", "both"], default="expression")
    parser.add_argument("--n-top-genes", type=int, default=5000, help="0 to use all genes")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 64])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--min-samples-per-class", type=int, default=20)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        data_source=args.data_source,
        n_top_genes=args.n_top_genes,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        min_samples_per_class=args.min_samples_per_class,
        val_split=args.val_split,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )
