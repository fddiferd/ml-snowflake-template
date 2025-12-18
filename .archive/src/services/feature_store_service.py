import logging
from datetime import datetime

from snowflake.snowpark import Session
from snowflake.snowpark.dataframe import DataFrame
from snowflake.ml.feature_store import FeatureStore, CreationMode, Entity, FeatureView # type: ignore

from src.utils.typing import get_version


logger = logging.getLogger(__name__)


class FeatureStoreService:
    def __init__(
        self, 
        session: Session,
        name: str,
        timestamp_col: str | None = None
    ):
        self.session = session
        self.name = name
        self.fs = self._get_feature_store()
        self.entities: list[Entity] = []
        self.feature_views: list[FeatureView] = []
        self.spine: DataFrame | None = None
        self.timestamp_col: str | None = timestamp_col

    # Public Methods
    def set_entity(self, join_keys: list[str], name: str | None = None, desc: str | None = None) -> None:
        name = name or f'{self.name}_ENTITY'
        entity = Entity(
            name=name,
            join_keys=[k.upper() for k in join_keys],
            desc=desc or f'Unique join keys for {name} feature views',
        )
        self.fs.register_entity(entity)
        logger.info(f"Entity {name} created")
        self.entities.append(entity)

    def set_feature_view(self, feature_df: DataFrame, version_number: int, name: str | None = None, refresh_freq: str | None = None, desc: str | None = None) -> None:
        name = name or f'{self.name}_FV'
        version = get_version(version_number)
        
        if len(self.entities) == 0:
            raise ValueError("No entities found")
        feature_view = FeatureView(
            name=name,
            entities=self.entities,
            feature_df=self._convert_timestamp_tz_columns(feature_df),
            refresh_freq=refresh_freq,
            desc=desc or f'{name} feature view',
        )
        registered_fv = self.fs.register_feature_view(feature_view, version=version, overwrite=True)
        logger.info(f"Feature view {name} created")
        self.feature_views.append(registered_fv)

    def set_spine(self, spine_df: DataFrame) -> None:
        self.spine = self._convert_timestamp_tz_columns(spine_df)
        logger.info(f"Spine {self.name} SPINE created")
        
    def get_dataset(self) -> DataFrame:
        if self.spine is None:
            raise ValueError("Spine is not set")
        if len(self.feature_views) == 0:
            raise ValueError("Feature views are not set")
        return self.fs.generate_dataset(
            name=f'{self.name}_DS',
            spine_df=self.spine,
            features=self.feature_views,
            version=f'{datetime.now().strftime("%Y%m%d%H%M%S")}',
            spine_timestamp_col=self.timestamp_col.upper() if self.timestamp_col else None
        ).read.to_snowpark_dataframe()

    # Helper Methods
    def _get_feature_store(self):
        return FeatureStore(
            session=self.session,
            database=self.session.get_current_database(),
            name=f'{self.name}_FS',
            default_warehouse=self.session.get_current_warehouse(),
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
    
    def _convert_timestamp_tz_columns(self, df: DataFrame) -> DataFrame:
        """Convert TIMESTAMP_TZ columns to TIMESTAMP to avoid Parquet export issues."""
        from snowflake.snowpark.functions import col, to_timestamp
        from snowflake.snowpark.types import TimestampType, TimestampTimeZone
        
        # Get schema information
        schema_fields = df.schema.fields
        converted_cols = []
        
        for field in schema_fields:
            # Check if it's a TimestampType with TZ timezone
            if (isinstance(field.datatype, TimestampType) and 
                field.datatype.tz == TimestampTimeZone.TZ):
                # Convert TIMESTAMP_TZ to TIMESTAMP (NTZ - no timezone)
                logger.info(f"Converting TIMESTAMP_TZ column '{field.name}' to TIMESTAMP")
                converted_cols.append(to_timestamp(col(field.name)).alias(field.name))
            else:
                # Keep other columns as-is
                converted_cols.append(col(field.name))
        
        if converted_cols:
            return df.select(*converted_cols)
        else:
            return df