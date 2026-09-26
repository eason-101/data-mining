"""One-command reproduction of the report: clean, cross-validate, refit, evaluate.

Python 3.12; use requirements.txt. Example:
    python train.py --data ../dataset.zip


dataset is not updated, so you need add a dataset here.
"""
import argparse
import hashlib
import io
import json
import platform
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

SEED = 50
METRICS = ["average_precision", "roc_auc", "f1", "balanced_accuracy"]


class AuditedLogisticRegression(LogisticRegression):
    def fit(self, X, y, sample_weight=None):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            result = super().fit(X, y, sample_weight=sample_weight)
        self.converged_ = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        for warning in caught:
            if not issubclass(warning.category, ConvergenceWarning):
                warnings.warn(str(warning.message), warning.category)
        return result


def converged(estimator, X, y):
    return float(getattr(estimator.named_steps["model"], "converged_", True))


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if hasattr(obj, "get_params"):
            return repr(obj)
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert), encoding="utf-8")


def load_data(path):
    frames, audit = {}, {}
    with zipfile.ZipFile(path) as archive:
        for split in ("train", "test"):
            names = [n for n in archive.namelist()
                     if n == f"{split}.csv" or n.endswith(f"/{split}.csv")]
            if len(names) != 1:
                raise ValueError(f"Expected exactly one {split}.csv in the archive")
            raw = archive.read(names[0])
            frame = pd.read_csv(io.BytesIO(raw))
            if "label" not in frame:
                raise ValueError("Missing label column")
            if not set(frame.label.dropna()).issubset({"yes", "no"}):
                raise ValueError("Unexpected target labels; inspect the source data")
            labelled = frame.dropna(subset=["label"]).copy()
            audit[split] = {
                "sha256": hashlib.sha256(raw).hexdigest(), "original_rows": len(frame),
                "removed_target_rows": frame.index[frame.label.isna()].tolist(),
                "labelled_rows": len(labelled),
                "class_counts": labelled.label.value_counts().to_dict(),
                "duplicate_rows": int(labelled.duplicated().sum()),
                "predictor_missing": labelled.drop(columns="label").isna().sum().to_dict(),
            }
            frames[split] = labelled
    train, test = frames["train"], frames["test"]
    if list(train.columns) != list(test.columns):
        raise ValueError("Train/test columns differ")
    X, Xt = train.drop(columns="label"), test.drop(columns="label")
    cat = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num = [c for c in X if c not in cat]
    q1, q3 = X[num].quantile(.25), X[num].quantile(.75)
    audit.update(numeric_columns=num, categorical_columns=cat,
        iqr_flags=((X[num] < q1 - 1.5 * (q3-q1)) | (X[num] > q3 + 1.5 * (q3-q1))).sum().to_dict(),
        numeric_correlations=X[num].corr().to_dict(),
        category_counts={c: X[c].value_counts().to_dict() for c in cat},
        unseen_test_categories={c: sorted(set(Xt[c].dropna()) - set(X[c].dropna())) for c in cat})
    return X, Xt, train.label.eq("yes").astype(int), test.label.eq("yes").astype(int), cat, num, audit


def pipeline_and_grid(name, cat, num):
    tree = name == "decision_tree"
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore" if tree else "infrequent_if_exist",
                                 **({"sparse_output": False} if tree else
                                    {"min_frequency": 20, "drop": "first"}))),
    ])
    steps = [("impute", SimpleImputer(strategy="median"))]
    if not tree:
        steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer([("categorical", categorical, cat),
                              ("numeric", Pipeline(steps), num)],
                             sparse_threshold=0 if tree else .3)
    model = (DecisionTreeClassifier(random_state=SEED) if tree else
             AuditedLogisticRegression(solver="liblinear", l1_ratio=0.,
                                      max_iter=2000, tol=1e-4, random_state=SEED))
    grid = {
        "preprocess__categorical__impute__strategy": ["most_frequent", "constant"],
        "preprocess__numeric__impute__strategy": ["median", "mean"],
        "model__class_weight": [None, "balanced"],
    }
    if tree:
        grid.update({"model__criterion": ["gini", "entropy"],
                     "model__max_depth": [3, 4, 5, 6],
                     "model__min_samples_leaf": [50, 100, 250]})
    else:
        grid.update({"preprocess__categorical__encode__min_frequency": [None, 20, 50],
                     "preprocess__categorical__encode__drop": [None, "first"],
                     "preprocess__numeric__scale": [StandardScaler(), "passthrough"],
                     "model__C": [.01, .1, 1., 10.], "model__l1_ratio": [0., 1.]})
    return Pipeline([("preprocess", prep), ("model", model)]), grid


def metrics(y, probability):
    pred = (probability >= .5).astype(int)
    return {"accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision_positive": precision_score(y, pred, zero_division=0),
            "recall_positive": recall_score(y, pred, zero_division=0),
            "f1_positive": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, probability),
            "average_precision": average_precision_score(y, probability),
            "confusion_matrix": confusion_matrix(y, pred).tolist()}


def comparisons(name, results, winner):
    params = results["params"]
    valid = [i for i in range(len(params)) if results["mean_test_converged"][i] == 1]
    rows = []
    for key in params[winner]:
        for value in dict.fromkeys(repr(p[key]) for p in params):
            indices = [i for i in valid if repr(params[i][key]) == value]
            if not indices:
                continue
            best = max(indices, key=lambda i: results["mean_test_average_precision"][i])
            matched = [i for i in indices if all(repr(v) == repr(params[winner][k])
                       for k, v in params[i].items() if k != key)]
            rows.append({"model": name, "parameter": key, "option": value,
                "selected": value == repr(params[winner][key]),
                "best_retuned_cv_ap": results["mean_test_average_precision"][best],
                "matched_cv_ap": results["mean_test_average_precision"][matched[0]] if matched else None})
    return rows


def save_explanations(name, pipe, X, out):
    prep, model = pipe.named_steps["preprocess"], pipe.named_steps["model"]
    names = prep.get_feature_names_out()
    if name == "decision_tree":
        (out / "tree_rules.txt").write_text(export_text(model, feature_names=list(names),
                                                      decimals=6), encoding="utf-8")
        row = prep.transform(X.iloc[:1])[0]
        path = []
        for node in model.decision_path(row.reshape(1, -1)).indices:
            j = model.tree_.feature[node]
            if j >= 0:
                threshold = model.tree_.threshold[node]
                path.append(f"{names[j]} = {row[j]:.6f} "
                            f"{'<=' if row[j] <= threshold else '>'} {threshold:.6f}")
        path.append(f"P(yes) = {pipe.predict_proba(X.iloc[:1])[0, 1]:.12f}")
        (out / "tree_example.txt").write_text("\n".join(path), encoding="utf-8")
        return {"depth": model.get_depth(), "leaves": model.get_n_leaves()}
    weights = model.coef_[0]
    pd.DataFrame({"feature": names, "coefficient": weights}).to_csv(out / "coefficients.csv", index=False)
    row = prep.transform(X.iloc[:1])
    row = row.toarray().ravel() if hasattr(row, "toarray") else np.asarray(row).ravel()
    contributions = row * weights
    expected = model.decision_function(row.reshape(1, -1))[0]
    assert np.isclose(contributions.sum() + model.intercept_[0], expected)
    pd.DataFrame({"feature": ["intercept", *names],
                  "log_odds_contribution": [model.intercept_[0], *contributions]}).to_csv(
                      out / "logistic_example.csv", index=False)
    return {"iterations": int(model.n_iter_[0]), "converged": model.converged_,
            "coefficients": len(weights), "nonzero_coefficients": int(np.count_nonzero(weights))}


def plot_curves(predictions, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.05))
    for name, label, color, style in [
            ("decision_tree", "Decision Tree", "#005791", "-"),
            ("logistic_regression", "Logistic Regression", "#bc4d00", "--")]:
        p, y = predictions[f"{name}_probability"], predictions.actual
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=label, color=color, ls=style, lw=1.2)
        axes[1].plot(recall, precision, label=label, color=color, ls=style, lw=1.2)
    axes[0].plot([0, 1], [0, 1], ":", color="gray", lw=.7, label="Baseline")
    axes[1].axhline(predictions.actual.mean(), ls=":", color="gray", lw=.7)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="(a) ROC", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="(b) Precision-recall", xlim=(0, 1), ylim=(.2, 1))
    for ax in axes:
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(pad=.6, rect=(0, 0, 1, .87))
    fig.savefig(output, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Original course dataset.zip")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    X, Xt, y, yt, cat, num, audit = load_data(args.data)
    write_json(args.output / "data_audit.json", audit)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    assignment = np.zeros(len(X), dtype=int)
    for i, (_, val) in enumerate(folds):
        assignment[val] = i
    pd.DataFrame({"original_train_row": X.index, "validation_fold": assignment}).to_csv(
        args.output / "folds.csv", index=False)
    results = {"protocol": {"seed": SEED, "folds": 5, "threshold": .5,
        "selection": "Maximum mean validation AP among fully converged candidates; exact ties use grid order",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__}}, "models": {}}
    predictions = pd.DataFrame({"original_test_row": Xt.index, "actual": yt.to_numpy()})
    evidence, summary = [], []
    for name in ("decision_tree", "logistic_regression"):
        start = time.perf_counter()
        pipe, grid = pipeline_and_grid(name, cat, num)
        search = GridSearchCV(pipe, grid, cv=folds, refit=False, n_jobs=args.jobs,
            scoring={**{m: m for m in METRICS}, "converged": converged},
            error_score="raise", return_train_score=False, verbose=1)
        search.fit(X, y)
        cv = search.cv_results_
        valid = np.flatnonzero(cv["mean_test_converged"] == 1)
        if not len(valid):
            raise RuntimeError(f"No converged {name} candidate")
        winner = int(valid[np.argmax(cv["mean_test_average_precision"][valid])])
        pd.DataFrame(cv).to_csv(args.output / f"{name}_cv.csv", index=False)
        evidence.extend(comparisons(name, cv, winner))
        pipe.set_params(**cv["params"][winner]).fit(X, y)
        if not converged(pipe, X, y):
            raise RuntimeError("Final refit failed to converge")
        probability = pipe.predict_proba(Xt)[:, 1]
        predictions[f"{name}_probability"] = probability
        predictions[f"{name}_prediction"] = (probability >= .5).astype(int)
        test_metrics = metrics(yt, probability)
        result = {"selected_params": cv["params"][winner], "candidates": len(cv["params"]),
            "eligible_candidates": len(valid), "cv_ap": cv["mean_test_average_precision"][winner],
            "cv_ap_sd": cv["std_test_average_precision"][winner],
            "fold_ap": [cv[f"split{i}_test_average_precision"][winner] for i in range(5)],
            "complexity": save_explanations(name, pipe, Xt, args.output),
            "train": metrics(y, pipe.predict_proba(X)[:, 1]), "test": test_metrics,
            "elapsed_seconds": time.perf_counter() - start}
        results["models"][name] = result
        summary.append({"model": name, **{k: v for k, v in test_metrics.items() if k != "confusion_matrix"}})
        print(name, "CV AP", result["cv_ap"], "test AP", test_metrics["average_precision"], flush=True)
        write_json(args.output / "results.json", results)
    baseline = metrics(yt, np.zeros(len(yt)))
    results["majority_baseline"] = baseline
    summary.append({"model": "majority_no", **{k: v for k, v in baseline.items() if k != "confusion_matrix"}})
    write_json(args.output / "results.json", results)
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(evidence).to_csv(args.output / "comparisons.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    plot_curves(predictions, args.output / "curves.png")
    print(f"Complete. Results saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
