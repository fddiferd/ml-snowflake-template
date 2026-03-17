import logging
import time
from snowflake.snowpark import Session

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv()

from src.writers import WriterType
from src.utils.slack import send_slack_notification
from projects.adwords_gclid_upload import get_session
from projects.adwords_gclid_upload.service import GclidUploadService


logger = logging.getLogger(__name__)


def main(
    session: Session | None = None,
    writer_type: WriterType = WriterType.SNOWFLAKE,
    reset_schema: bool = False,
):
    if session is None:
        session = get_session()

    start_time = time.time()

    send_slack_notification(
        session=session,
        header="Adwords GCLID Upload",
        text="GCLID upload pipeline started",
        is_success=True,
    )

    service = GclidUploadService(session)
    service.run()

    elapsed = time.time() - start_time
    try:
        send_slack_notification(
            session=session,
            header="Adwords GCLID Upload",
            text=f"GCLID upload pipeline completed successfully in {elapsed:.1f} seconds",
            is_success=True,
        )
    except Exception:
        logger.warning("Failed to send success notification to Slack")


if __name__ == "__main__":
    main()
