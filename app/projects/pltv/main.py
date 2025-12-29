import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv()

from projects.pltv import (
    Level,
    get_session,
    get_df,
    get_df_from_cache,
    clean_df,
    ModelService,
)


logger = logging.getLogger(__name__)


def main_level(level: Level, from_cache: bool = False, to_cache: bool = False):
    # get snowflake session
    session = get_session()
    # get dataset
    if from_cache:
        try:
            df = get_df_from_cache(level)
        except FileNotFoundError:
            logger.warning(f"Cache file not found, getting dataset from snowflake")
            df = get_df(session, level, save_to_cache=to_cache)
    else:
        df = get_df(session, level, save_to_cache=to_cache)
    # clean dataset
    clean_df(df)
    # run model service
    model_service = ModelService(
        session, 
        level, 
        df,
        test_train_split=True,
        save_to_db=False,
        save_to_cache=to_cache,
    )
    model_service.run()
        

def main(from_cache: bool = False, to_cache: bool = False):
    for level in Level:
        if level == Level.TRAFFIC_SOURCE:
            main_level(
                level, 
                from_cache, 
                to_cache
            )
            


if __name__ == "__main__":
    main(
        from_cache=True,
        to_cache=True
    )