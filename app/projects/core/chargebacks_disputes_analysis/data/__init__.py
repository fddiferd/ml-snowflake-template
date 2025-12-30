import os

from .adyen import queries as adyen_queries
from .braintree import queries as braintree_queries
from .worldpay import queries as worldpay_queries


CACHE_PATH = "app/projects/core/chargebacks_disputes_analysis/data/cache"

queries = (
    adyen_queries
    + braintree_queries
    # + worldpay_queries
)


def get_file_path(date: str, extension: str = "parquet") -> str:
    os.makedirs(CACHE_PATH, exist_ok=True)
    return os.path.join(CACHE_PATH, f"{date}.{extension}")