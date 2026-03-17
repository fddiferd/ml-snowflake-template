if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import logging
    logging.basicConfig(level=logging.INFO)


from snowflake.snowpark import Session
from projects import Project


def get_session() -> Session:
    """Get a Snowflake session configured for the VBB project."""
    from src.connection.session import get_session as get_snowflake_session
    return get_snowflake_session(Project.VBB)


if __name__ == "__main__":
    session = get_session()