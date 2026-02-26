from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation import evaluate_model, save_confusion_matrix, save_roc_pr_curves
from src.model import tune_model
from src.preprocessing import build_preprocessor, load_data
from src.utils import save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Telco churn XGBoost pipeline.")
    parser.add_argument(
        "--data-path",
        default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to input CSV dataset.",
    )
    parser.add_argument(
        "--model-out",
        default="models/xgb_churn_model.pkl",
        help="Output path for trained model artifact.",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/figures",
        help="Directory to save evaluation plots.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=40,
        help="Number of random search iterations (>=1).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Cross-validation folds for RandomizedSearchCV (>=2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_iter < 1:
        raise ValueError("--n-iter must be >= 1")
    if args.cv < 2:
        raise ValueError("--cv must be >= 2")

    X_train, X_test, y_train, y_test = load_data(args.data_path)
    preprocessor = build_preprocessor(X_train)

    model = tune_model(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        n_iter=args.n_iter,
        cv=args.cv,
    )

    proba, preds = evaluate_model(model, X_test, y_test)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(y_test, preds, figures_dir)
    save_roc_pr_curves(y_test, proba, figures_dir)

    save_model(model, args.model_out)


if __name__ == "__main__":
    main()
