import logging

from projects import Project
from src.connection.session import get_session


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_pltv_session():
    session = get_session(Project.PLTV)
    assert session is not None
    assert session.get_current_account() is not None
    assert session.get_current_user() is not None
    assert session.get_current_role() is not None
    assert session.get_current_database() is not None
    assert session.get_current_warehouse() is not None


if __name__ == "__main__":
    test_pltv_session()