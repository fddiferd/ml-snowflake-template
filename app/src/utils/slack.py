
NOTIFICATION_INTEGRATION_NAME = f'ml_layer_notifications'


def send_slack_notification(session, header: str, text: str, is_success: bool):
    emoji = "✅" if is_success else "❌"
    # Use Snowflake’s sanitizer for safety
    session.sql(f"""
        call system$send_snowflake_notification(
          snowflake.notification.TEXT_PLAIN(
            snowflake.notification.sanitize_webhook_content('{emoji} {header}: {text}')
          ),
          snowflake.notification.integration('{NOTIFICATION_INTEGRATION_NAME}')
        );
    """).collect()


"""
use database ml_layer;
use schema feature_store;

CREATE OR REPLACE SECRET ml_layer_slack_webhook_secret
  TYPE = GENERIC_STRING
  SECRET_STRING = 'T02GL1CBD/B09FSUGR7NK/QNUa56H4Pptyxn46OR9gIFl5';


CREATE OR REPLACE NOTIFICATION INTEGRATION ml_layer_notifications
  TYPE = WEBHOOK
  ENABLED = TRUE
  WEBHOOK_URL = 'https://hooks.slack.com/services/SNOWFLAKE_WEBHOOK_SECRET'
  WEBHOOK_SECRET = ml_layer_slack_webhook_secret
  WEBHOOK_BODY_TEMPLATE = '{
    "text": ":rotating_light: *Task Alert*",
    "blocks": [
      {
        "type": "header",
        "text": {
          "type": "plain_text",
          "text": "Task Notification",
          "emoji": true
        }
      },
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "SNOWFLAKE_WEBHOOK_MESSAGE"
        }
      },
      {
        "type": "context",
        "elements": [
          {
            "type": "mrkdwn",
            "text": "Sent from ML Layer"
          }
        ]
      }
    ]
  }'
  WEBHOOK_HEADERS = ('Content-Type'='application/json');


  CALL SYSTEM$SEND_SNOWFLAKE_NOTIFICATION(
  SNOWFLAKE.NOTIFICATION.TEXT_PLAIN(
    SNOWFLAKE.NOTIFICATION.SANITIZE_WEBHOOK_CONTENT(
      'Test notification: Snowflake-Slack integration is working! 🎉'
    )
  ),
  SNOWFLAKE.NOTIFICATION.INTEGRATION('ml_layer_notifications')
);
"""


if __name__ == "__main__":
  from dotenv import load_dotenv
  load_dotenv()
  from src.connection.session import get_session
  send_slack_notification(
    session=get_session(use_public_schema=True), 
    header="Core Test", 
    text="This is a success message", 
    is_success=True
  )