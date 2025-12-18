from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes
import os

from src.constants import PRIVATE_KEY_PEM_PATH

def get_private_key() -> bytes:
    try:
        private_key_pem = _get_private_key_pem_from_env()
    except ValueError:
        private_key_pem = _get_private_key_pem_from_path()
    return _get_private_key_bytes(private_key_pem)

def _get_private_key_pem_from_path() -> PrivateKeyTypes:
    def _get(path: str) -> PrivateKeyTypes:
        with open(path, "rb") as key_file:
            p_key: PrivateKeyTypes = serialization.load_pem_private_key(
                key_file.read(),
                password=None  # Key was generated with -nocrypt flag
            )
        return p_key
    try:
        return _get(PRIVATE_KEY_PEM_PATH)
    except FileNotFoundError:
        # go up one level and try again (for when running from app/ directory)
        return _get(f'../{PRIVATE_KEY_PEM_PATH}')

def _get_private_key_pem_from_env() -> PrivateKeyTypes:
    private_key_pem = os.getenv("PRIVATE_KEY_PEM")
    if private_key_pem is None:
        raise ValueError("PRIVATE_KEY_PEM is not set")
    return serialization.load_pem_private_key(
        private_key_pem.encode(),
        password=b"TCGmllayer1!"
    )

def _get_private_key_bytes(private_key_pem: PrivateKeyTypes) -> bytes:
    return private_key_pem.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
