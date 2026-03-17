"""
Stored Procedure Entry Points for VBB
======================================

Two separate handlers for Snowflake Tasks:
  - run_train_sproc: Weekly model training with diagnostics
  - run_predict_sproc: Daily prediction + write to table/GCS

Each handler follows the same pattern as create_sproc_handler but calls
a specific entry function in projects.vbb.main.
"""

from src.sproc import create_sproc_handler


run_train_sproc = create_sproc_handler(
    project_name="VBB_TRAIN",
    main_module_path="projects.vbb.main_train",
    target="PROD",
)

run_predict_sproc = create_sproc_handler(
    project_name="VBB_PREDICT",
    main_module_path="projects.vbb.main_predict",
    target="PROD",
)
