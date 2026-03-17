from src.initialize import load
load()

import os
import logging
import pandas as pd
from snowflake.snowpark import Session
from sklearn.model_selection import train_test_split

from projects.vbb.data.training import get_training_df
from projects.vbb.model.pipeline import train_and_evaluate
from projects.vbb.constants import TARGET_COL
from src.pipeline.xgboost import PRED_YHAT_COL
from src.utils.visualization import (
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_feature_importances,
    plot_decile_lift,
)


logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


class VBBModelService:

    def __init__(self, session: Session):
        self.session = session

    def run(self):
        logger.info("=== VBB Model Training ===")

        df = get_training_df(self.session)
        logger.info(f"Loaded {len(df):,} rows")

        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df = pd.DataFrame(train_df)
        test_df = pd.DataFrame(test_df)
        logger.info(f"Train: {len(train_df):,}  Test: {len(test_df):,}")

        result_df, model, evaluation, importance_df = train_and_evaluate(train_df, test_df)

        logger.info(f"RMSE:     {evaluation.rmse:.2f}")
        logger.info(f"MAE:      {evaluation.mae:.2f}")
        logger.info(f"MAPE:     {evaluation.mape:.1f}%")
        logger.info(f"R²:       {evaluation.r2:.4f}")
        logger.info(f"Spearman: {evaluation.spearman:.4f}")

        self._save_visualizations(result_df, importance_df)
        self._save_results(result_df, importance_df, evaluation)

        return result_df, model, evaluation, importance_df

    def _save_visualizations(self, result_df: pd.DataFrame, importance_df: pd.DataFrame):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        y_true = pd.Series(result_df[TARGET_COL])
        y_pred = pd.Series(result_df[PRED_YHAT_COL])

        fig = plot_actual_vs_predicted(y_true, y_pred)
        fig.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
        logger.info("Saved actual_vs_predicted.png")

        fig = plot_residual_distribution(y_true, y_pred)
        fig.savefig(os.path.join(OUTPUT_DIR, "residuals.png"), dpi=150)
        logger.info("Saved residuals.png")

        fig = plot_feature_importances(importance_df)
        fig.savefig(os.path.join(OUTPUT_DIR, "feature_importances.png"), dpi=150)
        logger.info("Saved feature_importances.png")

        fig = plot_decile_lift(y_true, y_pred)
        fig.savefig(os.path.join(OUTPUT_DIR, "decile_lift.png"), dpi=150)
        logger.info("Saved decile_lift.png")

    def _save_results(self, result_df, importance_df, evaluation):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        result_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
        importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)

        with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
            f.write(f"RMSE:     {evaluation.rmse:.2f}\n")
            f.write(f"MAE:      {evaluation.mae:.2f}\n")
            f.write(f"MAPE:     {evaluation.mape:.1f}%\n")
            f.write(f"R2:       {evaluation.r2:.4f}\n")
            f.write(f"Spearman: {evaluation.spearman:.4f}\n")

        logger.info("Saved results to output/")


if __name__ == "__main__":
    from projects.vbb import get_session
    session = get_session()
    service = VBBModelService(session)
    service.run()
