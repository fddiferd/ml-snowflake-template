import logging


logger = logging.getLogger(__name__)


def get_version(version_number: int) -> str:
    return f'V{version_number}'