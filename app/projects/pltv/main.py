from projects.pltv import (
    Level,
    get_session,
    get_df,
    get_df_from_cache,
    clean_df,
    ModelService,
)

def main(level: Level, from_cache: bool = False, to_cache: bool = False):

    # get snowflake session
    session = get_session()

    # get dataset
    if from_cache:
        df = get_df_from_cache()
    else:
        df = get_df(session, level, save_to_cache=to_cache)

    # clean dataset
    clean_df(df)

    # run model service
    model_service = ModelService(level, df)
    model_service.run()
        


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    
    level = Level(group_bys=["brand", "sku_type", "channel"])

    main(level, from_cache=True, to_cache=True)