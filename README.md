# ML Layer

A multi-project ML platform that runs Python pipelines inside Snowflake via stored procedures. Each project trains models and writes predictions to Snowflake tables and/or GCS stages, orchestrated by Snowflake Tasks on a schedule.

## Architecture

```
app/
├── projects/           # Independent ML projects
│   ├── vbb/            # Value Based Bidding (Google Ads conversion values)
│   ├── pltv/           # Predicted Lifetime Value
│   └── adwords_gclid_upload/  # GCLID conversion upload to Google Ads
├── src/                # Shared code across all projects
│   ├── pipeline/       # XGBoost pipeline, preprocessing, prediction intervals
│   ├── connection/     # Snowflake session factory
│   ├── environment.py  # DEV/STAGING/PROD environment config
│   ├── sproc/          # Stored procedure handler factory
│   ├── utils/          # Visualization, model evaluation, Slack notifications
│   └── writers/        # Data output writers
setup/
├── setup_infrastructure.sql  # Databases, warehouses, roles (run once)
├── {project}/
│   ├── tasks.sql             # Stored procedures + scheduled tasks
│   └── google_cloud_bucket.sql  # GCS integration (if applicable)
.github/workflows/
└── deploy-{project}.yml      # CI/CD per project
scripts/
└── deploy.sh                 # Manual deployment script
```

## How It Works

1. **Code lives in `app/`** -- each project has its own `data/`, `model/`, `main.py`, and `sproc.py`
2. **`deploy.sh` or GitHub Actions** packages `app/` into `ml_layer.zip`, uploads it to a Snowflake stage
3. **Snowflake stored procedures** import the zip and call the project's sproc handler
4. **Snowflake Tasks** run the procedures on a cron schedule
5. **Results** are written to Snowflake tables and optionally exported as CSV to GCS stages

```
GitHub push → Actions workflow → snow snowpark build → upload zip to stage
                                                     → CREATE PROCEDURE (handler = sproc.py)
                                                     → CREATE TASK (cron schedule)

Snowflake Task fires → calls stored procedure → imports ml_layer.zip
                                              → runs main.py (train/predict)
                                              → writes results to table/GCS
                                              → sends Slack notification
```

## Project Structure (each project follows this pattern)

```
app/projects/{project}/
├── __init__.py          # get_session() for this project's database
├── constants.py         # Column lists, hyperparams, table/stage names
├── data/
│   ├── query.py         # SQL query for data extraction
│   ├── training.py      # Load + cache training data
│   └── prediction.py    # Load + cache prediction data
├── model/
│   ├── pipeline.py      # Project-specific feature engineering + shared XGBoost pipeline
│   └── service.py       # Orchestrator: train(), predict(), write outputs
├── main.py              # Entry point with Slack notifications
├── sproc.py             # Snowflake stored procedure handler
└── README.md            # Project-specific documentation
```

## Environment

Three targets controlled by the `TARGET` env var:

- **DEV** -- uses schema `DEV_{DEVELOPER}`, enables local caching, saves diagnostic plots
- **STAGING** -- staging schema, no caching
- **PROD** -- production schema, writes to GCS stages, Slack notifications

Config lives in `.env` (local) and `.snowflake/config.toml` (Snowflake connection).

## Quick Start

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env  # set TARGET=DEV, DEVELOPER=yourname, USE_CACHE=TRUE

# Run a project locally
cd app
python -m projects.vbb.main_predict
python -m projects.vbb.model.service
```

## Deployment

```bash
# Manual deploy
./scripts/deploy.sh vbb prod
./scripts/deploy.sh pltv prod
./scripts/deploy.sh adwords_gclid_upload prod

# Or push to main -- GitHub Actions deploys automatically
```

## Adding a New Project

1. Add an enum member to `app/projects/__init__.py`
2. Create `app/projects/{project}/` following the structure above
3. Create `setup/{project}/tasks.sql` with stored procedures and tasks
4. Create `.github/workflows/deploy-{project}.yml` (copy an existing one)
5. Run `setup/setup_infrastructure.sql` to create the database (if new)

## Key Shared Components

- **`src/pipeline/xgboost.py`** -- Sklearn-compatible XGBoost wrapper with preprocessing (imputation, scaling, one-hot encoding) and prediction intervals
- **`src/sproc/base.py`** -- Factory for stored procedure handlers with standard error handling and JSON result formatting
- **`src/environment.py`** -- Lazy-loaded environment config (DEV/STAGING/PROD)
- **`src/utils/slack.py`** -- Slack notifications via Snowflake webhook integration
- **`src/utils/visualization.py`** -- Matplotlib plots (actual vs predicted, residuals, feature importance, decile lift)
