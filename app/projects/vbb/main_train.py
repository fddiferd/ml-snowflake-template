"""Weekly training entry point (called by VBB_TRAIN sproc)."""
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
        header="VBB Model Training",
        text="Weekly model training started",
        is_success=True,
    )

    service = VBBModelService(session)
    evaluation = service.train()

    elapsed = time.time() - start_time
    try:
        send_slack_notification(
            session=session,
            header="VBB Model Training",
            text=(
                f"Training completed in {elapsed:.0f}s -- "
                f"R2: {evaluation.r2:.4f}, Spearman: {evaluation.spearman:.4f}"
            ),
            is_success=True,
        )
    except Exception:
        logger.warning("Failed to send success notification to Slack")


if __name__ == "__main__":
    main()
