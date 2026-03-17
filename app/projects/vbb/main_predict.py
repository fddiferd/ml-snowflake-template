"""Daily prediction entry point (called by VBB_PREDICT sproc)."""
if __name__ == "__main__":
    from src.initialize import load
    load()

import logging
import time
from snowflake.snowpark import Session

from src.utils.slack import send_slack_notification
from projects.vbb import get_session
from projects.vbb.model.service import VBBModelService


logger = logging.getLogger(__name__)


def main(session: Session | None = None, **kwargs):
    if session is None:
        session = get_session()

    start_time = time.time()

    send_slack_notification(
        session=session,
        header="VBB Prediction",
        text="Daily prediction pipeline started",
        is_success=True,
    )

    service = VBBModelService(session)
    service.train()
    result_df = service.predict()

    elapsed = time.time() - start_time
    try:
        send_slack_notification(
            session=session,
            header="VBB Prediction",
            text=f"Prediction completed in {elapsed:.0f}s -- {len(result_df):,} rows exported",
            is_success=True,
        )
    except Exception:
        logger.warning("Failed to send success notification to Slack")


if __name__ == "__main__":
    main()
