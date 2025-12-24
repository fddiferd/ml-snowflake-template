from projects.pltv.objects import Level


def main(level: Level, from_cache: bool = False, to_cache: bool = False):

    from projects.pltv.session import get_session
    session = get_session()

    if from_cache:
        from projects.pltv.dataset import get_df_from_cache
        df = get_df_from_cache()
    else:
        from projects.pltv.dataset import get_df
        df = get_df(session, level, save_to_cache=to_cache)

    from projects.pltv.feature_engineering import clean_df
    clean_df(df)

    from projects.pltv.model import run
    run(level, df, test=True)
        


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    
    level = Level(group_bys=["brand", "sku_type", "channel"])

    main(level, from_cache=True, to_cache=True)