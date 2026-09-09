import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import MLP, TwoHeadMLP

EXPRESSION_PATH = "Expression_(Short-read)_Public_26Q1_subsetted.csv"
MUTATION_PATH = "Damaging_Mutations_(Public_26Q1)_subsetted.csv"
SUBTYPE_PATH = "Subtype_Matrix_Public_26Q1_subsetted.csv"


def r2_per_gene(y_true, y_pred, min_variance=1e-8):
    """Vectorized R² per column - equivalent to sklearn.r2_score per column
    but fast enough for thousands of target genes, and guards against genes
    with near-zero variance within a given split, which otherwise blow up
    the ratio (e.g. an R² of -1e12) rather than just being noisy."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    var = y_true.var(axis=0)
    return np.where(var < min_variance, np.nan, 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot))


def pearson_per_gene(y_true, y_pred, min_variance=1e-8):
    """Vectorized Pearson correlation per column, with the same near-zero-
    variance guard as r2_per_gene (avoids scipy's ConstantInputWarning spam
    and returns a clean NaN instead)."""
    yt = y_true - y_true.mean(axis=0)
    yp = y_pred - y_pred.mean(axis=0)
    denom = np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0))
    var = y_true.var(axis=0)
    return np.where((var < min_variance) | (denom == 0), np.nan, (yt * yp).sum(axis=0) / np.where(denom == 0, np.nan, denom))


def select_mutation_genes_univariate(
    mutation_data, expression_data, train_idx, n_mutation_genes, min_mutation_count=10, score_agg="mean",
):
    """Score each candidate mutation gene by how much its mutation status
    actually explains target-gene expression, instead of just how often it's
    mutated. Computed on the training rows only so this is a genuine feature
    selection step and not information leaked from the validation split.

    score_agg controls how each gene's per-target R² values (squared
    correlation) are collapsed into one ranking score:
    - "mean": favors genes with a small, broad effect across many target
      genes. Can bury a gene whose effect is real but narrow (e.g. it
      strongly drives only a handful of specific downstream genes), since
      that gene's average across thousands of mostly-unaffected targets is
      small even though its peak effect is large.
    - "max": favors genes with a strong effect on at least one target gene,
      regardless of how many targets they influence - surfaces the
      narrow-but-strong genes "mean" discards.

    min_mutation_count filters out genes mutated in very few training
    samples first - correlations from a handful of carriers are dominated by
    noise (same reasoning as the min_variance guard in r2_per_gene) and would
    otherwise crowd out genuinely informative genes at the top of the ranking.
    """
    if score_agg not in ("mean", "max"):
        raise ValueError(f"score_agg must be 'mean' or 'max', got {score_agg!r}")

    mutation_train = mutation_data.iloc[train_idx]
    candidate_mask = (mutation_train > 0).sum(axis=0) >= min_mutation_count
    candidates = mutation_train.columns[candidate_mask]
    if len(candidates) == 0:
        raise ValueError(f"No mutation genes have >= {min_mutation_count} mutated samples in the training split.")

    Xc = mutation_train[candidates].values.astype(np.float64)
    Yc = expression_data.iloc[train_idx].values.astype(np.float64)

    Xz = (Xc - Xc.mean(axis=0)) / np.where(Xc.std(axis=0) == 0, np.nan, Xc.std(axis=0))
    Yz = (Yc - Yc.mean(axis=0)) / np.where(Yc.std(axis=0) == 0, np.nan, Yc.std(axis=0))
    Xz, Yz = np.nan_to_num(Xz), np.nan_to_num(Yz)

    corr = (Xz.T @ Yz) / Xz.shape[0]  # (n_candidates, n_target_genes)
    r2 = corr ** 2
    score = r2.mean(axis=1) if score_agg == "mean" else r2.max(axis=1)

    ranked = candidates[np.argsort(-score)]
    if n_mutation_genes and n_mutation_genes < len(ranked):
        return ranked[:n_mutation_genes]
    if n_mutation_genes and n_mutation_genes > len(ranked):
        print(
            f"Only {len(ranked)} mutation genes pass min_mutation_count={min_mutation_count} "
            f"(requested {n_mutation_genes}); using all of them."
        )
    return ranked


def load_dataset(
    n_mutation_genes, n_target_genes, input_source="both",
    mutation_selection="frequency", min_mutation_count=10, mutation_score_agg="mean", val_split=0.2, seed=0,
):
    """input_source: "both" (mutation + subtype, default), "mutation", or
    "subtype" - restricting to one block is useful for ablations, e.g.
    checking how much of the model's performance comes from subtype flags
    alone vs. genuine mutation->expression signal.

    mutation_selection: "frequency" (default - most frequently mutated genes,
    independent of the targets) or "univariate" (select genes whose mutation
    status is actually associated with target expression - see
    select_mutation_genes_univariate, including what mutation_score_agg
    does). val_split/seed are only used to carve out the same training rows
    main() will train on, so the univariate score never sees validation data.
    """
    if input_source not in ("both", "mutation", "subtype"):
        raise ValueError(f"input_source must be 'both', 'mutation', or 'subtype', got {input_source!r}")
    if mutation_selection not in ("frequency", "univariate"):
        raise ValueError(f"mutation_selection must be 'frequency' or 'univariate', got {mutation_selection!r}")

    mutation_data = pd.read_csv(MUTATION_PATH, index_col=0)
    subtype_data = pd.read_csv(SUBTYPE_PATH, index_col=0)
    expression_data = pd.read_csv(EXPRESSION_PATH, index_col=0)

    common_index = mutation_data.index.intersection(subtype_data.index).intersection(expression_data.index)
    mutation_data = mutation_data.loc[common_index]
    subtype_data = subtype_data.loc[common_index]
    expression_data = expression_data.loc[common_index]

    # Target: the most variable expression genes - the ones that actually
    # carry predictable signal, same reasoning as clustering on the most
    # variable genes in explore.ipynb. Selected before the mutation genes
    # since "univariate" mode needs the final target set to score against.
    expression_data = expression_data.loc[:, expression_data.var(axis=0) > 0]
    gene_variance = expression_data.var(axis=0).sort_values(ascending=False)
    target_genes = gene_variance.head(n_target_genes).index if n_target_genes else gene_variance.index
    expression_data = expression_data[target_genes]

    # Input: informative subset of mutated genes (same "informative subset"
    # idea as predict_mutations.py, just used as an input block here) plus
    # every subtype-matrix flag - the subtype matrix is compact enough to use
    # as-is. Skipped entirely when mutation data isn't part of the input.

    # mutation_data = mutation_data.loc[:, mutation_data.var(axis=0) > 0]
    if input_source in ("both", "mutation") and n_mutation_genes:
        if mutation_selection == "frequency":
            candidate_mask = (mutation_data > 0).sum(axis=0) >= min_mutation_count
            candidates = mutation_data.columns[candidate_mask]
            if len(candidates) == 0:
                raise ValueError(f"No mutation genes have >= {min_mutation_count} mutated samples in the training split.")
            if len(candidates) < n_mutation_genes:
                print(
                    f"Only {len(candidates)} mutation genes pass min_mutation_count={min_mutation_count} "
                    f"(requested {n_mutation_genes}); using all of them."
                )
                selected_mutated = candidates
            else:
                selected_mutated = (mutation_data > 0).mean(axis=0).sort_values(ascending=False).head(n_mutation_genes).index
            
        else:
            train_idx, _ = train_test_split(np.arange(len(common_index)), test_size=val_split, random_state=seed)
            selected_mutated = select_mutation_genes_univariate(
                mutation_data, expression_data, train_idx, n_mutation_genes,
                min_mutation_count=min_mutation_count, score_agg=mutation_score_agg,
            )
        mutation_data = mutation_data[selected_mutated]
    subtype_data = subtype_data.loc[:, subtype_data.var(axis=0) > 0]

    # Mutation gene columns and subtype-flag columns can collide (e.g. both
    # can have a "KRAS" column with a different meaning) - suffix to keep
    # them distinct once concatenated.
    if input_source == "both":
        X = pd.concat([mutation_data.add_suffix("_mut"), subtype_data.add_suffix("_subtype")], axis=1)
    elif input_source == "mutation":
        X = mutation_data.add_suffix("_mut")
    else:
        X = subtype_data.add_suffix("_subtype")

    return X.values.astype(np.float32), expression_data.values.astype(np.float32), list(X.columns), list(target_genes)


def train_loop(model, train_loader, val_loader, optimizer, criterion, epochs, patience, device, extra_loss_fn=None, log_prefix=""):
    """Standard train/eval loop with early stopping (restores the best-val_loss
    checkpoint). Shared by the main training run and, when pretrain_subtype is
    used, the subtype-only pretraining stage - both need identical logic, just
    on different models/data.

    extra_loss_fn, if given, is called with no arguments each training step
    and added to the MSE loss (used for the mutation-branch L1/group-lasso
    penalty) - it must read current parameters itself (e.g. via a closure)
    since it's re-evaluated every step as the weights change.
    """
    best_val_loss = float("inf")
    best_epoch = None
    best_state = None
    best_preds, best_targets = None, None
    epochs_without_improvement = 0

    history = []
    final_epoch = 0
    for epoch in range(1, epochs + 1):
        final_epoch = epoch
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        all_preds, all_targets = np.concatenate(all_preds), np.concatenate(all_targets)

        # Macro-averaged R² across target genes.
        val_r2 = float(np.nanmean(r2_per_gene(all_targets, all_preds)))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_r2": val_r2})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_preds, best_targets = all_preds, all_targets
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"{log_prefix}epoch {epoch:3d}/{epochs}  train_loss {train_loss:.4f}  "
            f"val_loss {val_loss:.4f}  val_r2 {val_r2:.3f}"
            + ("  (best)" if epochs_without_improvement == 0 else f"  ({epochs_without_improvement}/{patience} without improvement)")
        )

        if epochs_without_improvement >= patience:
            print(f"{log_prefix}Early stopping: no val_loss improvement in {patience} epochs. Restoring epoch {best_epoch}.")
            break

    model.load_state_dict(best_state)
    return dict(
        history=history, best_epoch=best_epoch, best_val_loss=best_val_loss,
        best_preds=best_preds, best_targets=best_targets, stopped_early=final_epoch < epochs,
    )


def transplant_subtype_pretrain(model, pretrain_model, hidden_dims_subtype):
    """Copy a pretrained subtype-only MLP's weights into a TwoHeadMLP's
    subtype branch and shared trunk, zeroing the shared trunk's
    mutation-derived input columns. This makes the two_head model start
    fine-tuning mathematically identical to the pretrained subtype-only
    model - the mutation branch begins with exactly zero effect on the
    output and can only start influencing predictions if gradient descent
    finds it useful, rather than both branches co-adapting from scratch
    (where the mutation branch's early noise can disturb an
    already-good-enough subtype pathway before it's converged).

    Requires pretrain_model's hidden_dims to equal
    list(hidden_dims_subtype) + list(hidden_dims_shared) - see the
    pretrain_subtype block in main() for how it's built.
    """
    pretrain_layers = [m for m in pretrain_model.network if isinstance(m, (nn.Linear, nn.BatchNorm1d))]
    idx = 0

    if hidden_dims_subtype:
        for m in model.branch_b:
            if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
                m.load_state_dict(pretrain_layers[idx].state_dict())
                idx += 1

    shared_modules = [m for m in model.shared if isinstance(m, (nn.Linear, nn.BatchNorm1d))]
    first_shared_linear, pretrain_first_shared_linear = shared_modules[0], pretrain_layers[idx]
    idx += 1
    with torch.no_grad():
        first_shared_linear.weight.zero_()
        first_shared_linear.weight[:, model.out_dim_a:] = pretrain_first_shared_linear.weight
        first_shared_linear.bias.copy_(pretrain_first_shared_linear.bias)

    for m in shared_modules[1:]:
        m.load_state_dict(pretrain_layers[idx].state_dict())
        idx += 1


def main(
    n_mutation_genes=1000,
    n_target_genes=100,
    input_source="both",
    mutation_selection="frequency",
    min_mutation_count=10,
    mutation_score_agg="mean",
    architecture="single",
    hidden_dims=(512, 256),
    hidden_dims_mutation=(512, 256),
    hidden_dims_subtype=(128,),
    dropout=0.3,
    weight_decay=1e-4,
    patience=10,
    pretrain_subtype=False,
    pretrain_epochs=200,
    pretrain_patience=20,
    l1_mutation=0.0,
    val_split=0.2,
    batch_size=128,
    lr=1e-3,
    epochs=50,
    seed=0,
    device=None,
    output_dir="tests",
):
    if architecture not in ("single", "two_head"):
        raise ValueError(f"architecture must be 'single' or 'two_head', got {architecture!r}")
    if architecture == "two_head" and input_source != "both":
        raise ValueError("architecture='two_head' needs both input blocks - set input_source='both'")
    if pretrain_subtype and architecture != "two_head":
        raise ValueError("pretrain_subtype needs architecture='two_head'")
    if l1_mutation and architecture != "two_head":
        raise ValueError("l1_mutation needs architecture='two_head'")
    if l1_mutation and not hidden_dims_mutation:
        raise ValueError("l1_mutation needs a non-empty hidden_dims_mutation (no first layer to penalize otherwise)")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device == "cuda" else ""))

    run_dir = Path("runs") / Path(output_dir) / f"predict_gene_expression_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    X, Y, input_features, target_genes = load_dataset(
        n_mutation_genes, n_target_genes, input_source=input_source,
        mutation_selection=mutation_selection, min_mutation_count=min_mutation_count,
        mutation_score_agg=mutation_score_agg, val_split=val_split, seed=seed,
    )
    print(f"{X.shape[0]} samples, {X.shape[1]} input features ({input_source}, {mutation_selection}/{mutation_score_agg}), {Y.shape[1]} target genes")

    config = dict(
        n_mutation_genes=n_mutation_genes, n_target_genes=n_target_genes, input_source=input_source,
        mutation_selection=mutation_selection, min_mutation_count=min_mutation_count, mutation_score_agg=mutation_score_agg,
        architecture=architecture, hidden_dims=list(hidden_dims),
        hidden_dims_mutation=list(hidden_dims_mutation), hidden_dims_subtype=list(hidden_dims_subtype),
        dropout=dropout, weight_decay=weight_decay, patience=patience,
        pretrain_subtype=pretrain_subtype, pretrain_epochs=pretrain_epochs, pretrain_patience=pretrain_patience,
        l1_mutation=l1_mutation,
        val_split=val_split, batch_size=batch_size, lr=lr, epochs=epochs, seed=seed,
        n_samples=X.shape[0], n_input_features=X.shape[1], n_target_genes_actual=Y.shape[1],
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "target_genes.json").write_text(json.dumps(target_genes, indent=2))
    (run_dir / "input_features.json").write_text(json.dumps(input_features, indent=2))

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_split, random_state=seed)

    input_scaler = StandardScaler().fit(X_train)
    X_train = input_scaler.transform(X_train)
    X_val = input_scaler.transform(X_val)

    # Expression scale varies a lot gene-to-gene; z-score the targets (fit on
    # the training split only) so the shared regression loss weighs every
    # target gene comparably, same reasoning as z-scoring for the heatmaps.
    target_scaler = StandardScaler().fit(Y_train)
    Y_train = target_scaler.transform(Y_train)
    Y_val = target_scaler.transform(Y_val)

    # The whole dataset is only a few MB - load it onto the device once so
    # batching doesn't pay a host-to-device transfer cost every step.
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

    criterion = nn.MSELoss()

    if architecture == "two_head":
        # load_dataset() always concatenates mutation columns before subtype
        # columns for input_source="both" - split point is just their count.
        n_mutation_features = sum(f.endswith("_mut") for f in input_features)
        n_subtype_features = sum(f.endswith("_subtype") for f in input_features)
        config["n_mutation_input_features"] = n_mutation_features
        config["n_subtype_input_features"] = n_subtype_features
        model = TwoHeadMLP(
            input_dim_a=n_mutation_features,
            input_dim_b=n_subtype_features,
            hidden_dims_a=list(hidden_dims_mutation),
            hidden_dims_b=list(hidden_dims_subtype),
            hidden_dims_shared=list(hidden_dims),
            num_classes=Y.shape[1],
            dropout=dropout,
        ).to(device)

        if pretrain_subtype:
            # Pretrain a plain MLP shaped exactly like "subtype branch, then
            # shared trunk" (hidden_dims_subtype + hidden_dims) on subtype-
            # only input, then transplant its weights into the two_head
            # model - see transplant_subtype_pretrain() for how the shapes
            # line up. Scaling is per-column, so slicing the already-fit
            # X_train/X_val at the subtype columns is equivalent to fitting
            # a scaler on subtype-only data directly.
            print(f"--- Pretraining subtype-only model ({pretrain_epochs} epochs, patience {pretrain_patience}) ---")
            pretrain_model = MLP(
                input_dim=n_subtype_features,
                hidden_dims=list(hidden_dims_subtype) + list(hidden_dims),
                num_classes=Y.shape[1],
                dropout=dropout,
            ).to(device)
            pretrain_train_loader = DataLoader(
                TensorDataset(X_train_t[:, n_mutation_features:], Y_train_t),
                batch_size=batch_size, shuffle=True, drop_last=True,
            )
            pretrain_val_loader = DataLoader(TensorDataset(X_val_t[:, n_mutation_features:], Y_val_t), batch_size=batch_size)
            pretrain_optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=lr, weight_decay=weight_decay)
            pretrain_result = train_loop(
                pretrain_model, pretrain_train_loader, pretrain_val_loader, pretrain_optimizer, criterion,
                pretrain_epochs, pretrain_patience, device, log_prefix="[pretrain] ",
            )
            pd.DataFrame(pretrain_result["history"]).to_csv(run_dir / "pretrain_history.csv", index=False)
            config["pretrain_best_epoch"] = pretrain_result["best_epoch"]
            config["pretrain_val_r2"] = float(np.nanmean(r2_per_gene(pretrain_result["best_targets"], pretrain_result["best_preds"])))
            print(
                f"--- Pretraining done: best epoch {pretrain_result['best_epoch']}, "
                f"val_r2={config['pretrain_val_r2']:.4f} - transplanting into two_head model ---"
            )
            transplant_subtype_pretrain(model, pretrain_model, list(hidden_dims_subtype))
    else:
        model = MLP(
            input_dim=X.shape[1],
            hidden_dims=list(hidden_dims),
            num_classes=Y.shape[1],  # one regression output per target gene
            dropout=dropout,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    extra_loss_fn = None
    if l1_mutation:
        first_layer_a = model.first_layer_a()
        # Group lasso, not plain elementwise L1: penalizing each mutation
        # feature's whole first-layer weight column (its L2 norm) is what
        # actually drives an unhelpful gene's total influence to zero,
        # rather than just sparsifying individual weights within it. This is
        # the same per-feature norm results.ipynb already uses for weight-
        # based importance, so a gene pruned here shows up there as ~0 too.
        def extra_loss_fn():
            return l1_mutation * first_layer_a.weight.norm(dim=0).sum()

    result = train_loop(model, train_loader, val_loader, optimizer, criterion, epochs, patience, device, extra_loss_fn=extra_loss_fn)
    best_epoch, best_val_loss = result["best_epoch"], result["best_val_loss"]
    best_preds, best_targets = result["best_preds"], result["best_targets"]

    config["best_epoch"] = best_epoch
    config["stopped_early"] = result["stopped_early"]
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    pd.DataFrame(result["history"]).to_csv(run_dir / "history.csv", index=False)
    torch.save(model.state_dict(), run_dir / "model.pt")
    np.savez(run_dir / "input_scaler.npz", mean=input_scaler.mean_, scale=input_scaler.scale_)
    np.savez(run_dir / "target_scaler.npz", mean=target_scaler.mean_, scale=target_scaler.scale_)

    # Per-gene validation performance (from the best epoch, matching the
    # restored checkpoint): R² and Pearson correlation, since a single
    # aggregate number hides which genes are actually predictable.
    per_gene = pd.DataFrame({
        "gene": target_genes,
        "r2": r2_per_gene(best_targets, best_preds),
        "pearson_r": pearson_per_gene(best_targets, best_preds),
    }).sort_values("r2", ascending=False)
    per_gene.to_csv(run_dir / "val_per_gene_metrics.csv", index=False)

    print(f"Saved run artifacts to {run_dir} (best epoch: {best_epoch})")
    print(f"  -> val_loss={best_val_loss:.4f}  val_r2={float(np.nanmean(per_gene['r2'])):.4f}")
    return model, input_scaler, target_scaler, target_genes, run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an MLP to predict gene expression from mutation status and subtype."
    )
    parser.add_argument("--n-mutation-genes", type=int, default=10000, help="0 to use all mutated genes")
    parser.add_argument("--n-target-genes", type=int, default=4000, help="most variable expression genes to predict; 0 to use all")
    parser.add_argument(
        "--input-source", type=str, default="both", choices=["both", "mutation", "subtype"],
        help="which input block(s) to use - restrict to one for an ablation",
    )
    parser.add_argument(
        "--mutation-selection", type=str, default="frequency", choices=["frequency", "univariate"],
        help="'frequency' picks the most-mutated genes; 'univariate' picks genes whose mutation status "
        "is actually associated with target expression (scored on the training split only)",
    )
    parser.add_argument(
        "--min-mutation-count", type=int, default=5,
        help="univariate selection only: drop candidate genes mutated in fewer than this many training samples",
    )
    parser.add_argument(
        "--mutation-score-agg", type=str, default="mean", choices=["mean", "max"],
        help="univariate selection only: 'mean' R² across targets favors broad-but-weak mutation genes; "
        "'max' favors genes with a strong effect on at least one target, even if narrow. Only changes anything "
        "when --n-mutation-genes is below the number of genes passing --min-mutation-count.",
    )
    parser.add_argument(
        "--architecture", type=str, default="single", choices=["single", "two_head"],
        help="'single' concatenates mutation+subtype into one MLP input; 'two_head' gives each block "
        "its own tower before merging, so a large mutation block can't drown out subtype (needs --input-source both)",
    )
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[1024, 512, 256, 128], help="shared trunk (two_head) or the whole network (single)")
    parser.add_argument("--hidden-dims-mutation", type=int, nargs="+", default=[512, 256], help="two_head only: mutation branch")
    parser.add_argument("--hidden-dims-subtype", type=int, nargs="+", default=[128], help="two_head only: subtype branch")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20, help="stop if val_loss doesn't improve for this many epochs")
    parser.add_argument(
        "--pretrain-subtype", action="store_true",
        help="two_head only: pretrain the subtype branch + shared trunk on subtype-only data first, then add the "
        "mutation branch and fine-tune - fine-tuning starts mathematically identical to the pretrained subtype-only "
        "model, so the mutation branch can only help, not disturb an already-converged subtype pathway",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=200, help="--pretrain-subtype only")
    parser.add_argument("--pretrain-patience", type=int, default=20, help="--pretrain-subtype only")
    parser.add_argument(
        "--l1-mutation", type=float, default=0.0,
        help="two_head only: group-lasso penalty strength on the mutation branch's first-layer weights (each "
        "feature's weight-column L2 norm, summed) - drives unhelpful mutation genes' influence toward zero during "
        "training instead of needing separate add/remove-and-retrain runs",
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="tests")
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu - default auto-detects")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        n_mutation_genes=args.n_mutation_genes,
        n_target_genes=args.n_target_genes,
        input_source=args.input_source,
        mutation_selection=args.mutation_selection,
        min_mutation_count=args.min_mutation_count,
        mutation_score_agg=args.mutation_score_agg,
        architecture=args.architecture,
        hidden_dims=args.hidden_dims,
        hidden_dims_mutation=args.hidden_dims_mutation,
        hidden_dims_subtype=args.hidden_dims_subtype,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        patience=args.patience,
        pretrain_subtype=args.pretrain_subtype,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_patience=args.pretrain_patience,
        l1_mutation=args.l1_mutation,
        val_split=args.val_split,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
    )
