from src.initialize import load
load()

import os
import logging
import pandas as pd
from snowflake.snowpark import Session, DataFrame as SnowparkDataFrame
from sklearn.model_selection import train_test_split

from src.pipeline.xgboost import PRED_YHAT_COL, PRED_YHAT_LOWER_COL, PRED_YHAT_UPPER_COL
from src.environment import environment as env
from src.utils.visualization import (
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_feature_importances,
    plot_decile_lift,
)
from projects.vbb.data.training import get_training_df
from projects.vbb.data.prediction import get_prediction_dfs
from projects.vbb.model.pipeline import train_and_evaluate, predict_only
from projects.vbb.constants import (
    TARGET_COL,
    BRAND_COL,
    PREDICTION_RESULTS_TABLE_NAME,
    GCS_STAGE_NAME,
    GCS_FILE_NAME,
    PASSTHROUGH_COLS,
)


logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

PREDICTION_OUTPUT_COLS = PASSTHROUGH_COLS + [BRAND_COL, PRED_YHAT_COL]


class VBBModelService:

    def __init__(self, session: Session):
        self.session = session
        self._training_df: pd.DataFrame | None = None

    def train(self):
        """Load training data, train model with evaluation. Saves diagnostics in DEV."""
        logger.info("=== VBB Model Training ===")

        df = get_training_df(self.session)
        self._training_df = df
        logger.info(f"Loaded {len(df):,} training rows")

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

        if env.target.is_dev:
            self._save_visualizations(result_df, importance_df)
            self._save_results(result_df, importance_df, evaluation)

        return evaluation

    def predict(self):
        """Load prediction data, run model, write results to Snowflake table + GCS stage."""
        logger.info("=== VBB Prediction ===")

        if self._training_df is None:
            logger.info("No cached training data -- loading for model training")
            self._training_df = get_training_df(self.session)

        prediction_df, canceled_df = get_prediction_dfs(self.session)
        logger.info(f"Active: {len(prediction_df):,}  Canceled: {len(canceled_df):,}")

        result_df, _ = predict_only(self._training_df, prediction_df)

        active_output = result_df.copy()
        canceled_output = canceled_df.copy()
        canceled_output[PRED_YHAT_COL] = 0.0
        canceled_output[PRED_YHAT_LOWER_COL] = 0.0
        canceled_output[PRED_YHAT_UPPER_COL] = 0.0

        combined = pd.DataFrame(pd.concat([active_output, canceled_output], ignore_index=True))
        logger.info(f"Total prediction output: {len(combined):,} rows")

        self._save_to_table(combined)
        if not env.target.is_dev:
            self._export_brand_files_to_gcs(pd.DataFrame(combined[PREDICTION_OUTPUT_COLS]))
        else:
            logger.info("Skipping GCS export (DEV mode)")

        return combined

    # ------------------------------------------------------------------
    # Snowflake table output
    # ------------------------------------------------------------------

    def _save_to_table(self, df: pd.DataFrame) -> None:
        logger.info(f"Appending {len(df):,} rows to {PREDICTION_RESULTS_TABLE_NAME}")
        self.session.write_pandas(
            df,
            PREDICTION_RESULTS_TABLE_NAME,
            auto_create_table=True,
            overwrite=False,
            use_logical_type=True,
        )

    # ------------------------------------------------------------------
    # GCS stage output (per-brand CSV)
    # ------------------------------------------------------------------

    def _export_brand_files_to_gcs(self, df: pd.DataFrame) -> None:
        unique_brands = df[BRAND_COL].dropna().unique().tolist()
        logger.info(f"Exporting {len(unique_brands)} brand files to GCS stage")
        for brand in unique_brands:
            brand_df = pd.DataFrame(df[df[BRAND_COL] == brand])
            self._overwrite_stage_file(brand_df, str(brand))

    def _overwrite_stage_file(self, df: pd.DataFrame, brand: str) -> None:
        sf_df: SnowparkDataFrame = self.session.create_dataframe(df)
        file_name = f"@{GCS_STAGE_NAME}/{GCS_FILE_NAME}_{brand}.csv"
        logger.info(f"Writing CSV to GCS stage: {file_name}")
        sf_df.write.copy_into_location(
            file_name,
            header=True,
            overwrite=True,  # type: ignore[arg-type]
            single=True,  # type: ignore[arg-type]
            file_format_type="CSV",
            format_type_options={
                "COMPRESSION": "NONE",
                "FIELD_OPTIONALLY_ENCLOSED_BY": '"',
            },
        )
        logger.info(f"Uploaded {file_name}")

    # ------------------------------------------------------------------
    # Diagnostic output (local files, used during train)
    # ------------------------------------------------------------------

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
    service.train()
    service.predict()
