import os
import logging
from enum import Enum


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)


# MARK: - Private Functions
def _get_var(var_name: str) -> str:
    var = os.getenv(var_name)
    if var is None:
        raise ValueError(f"{var_name} is not set")
    return var

def _get_optional_var(var_name: str) -> str | None:
    return os.getenv(var_name)

# MARK: - Enums
class Target(Enum):
    DEV = "DEV"
    STAGING = "STAGING"
    PROD = "PROD"

    @property
    def is_dev(self) -> bool:
        return self == Target.DEV
    
    @property
    def is_staging(self) -> bool:
        return self == Target.STAGING
    
    @property
    def is_prod(self) -> bool:
        return self == Target.PROD

# MARK: - Environment Class
class Environment:
    def __init__(self):
        self.target: Target = Target(_get_var("TARGET").upper())
        self.developer: str | None = _get_optional_var("DEVELOPER")

        self._validate_developer()
        self._log()

    @property
    def schema_name(self) -> str:
        if self.target.is_dev and self.developer is not None:
            return f"{self.target.value.upper()}_{self.developer.upper()}"
        return self.target.value.upper()

    def _validate_developer(self) -> None:
        if self.target.is_dev and self.developer is None:
            raise ValueError("DEVELOPER is not set for DEV target")

    def _log(self) -> None:
        logger.info(f"Target: {self.target.value}")
        if self.target.is_dev:
            logger.info(f"Developer: {self.developer}")
        logger.info(f"Schema Name: {self.schema_name}")

environment = Environment()