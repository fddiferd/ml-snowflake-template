
def load():
    from dotenv import load_dotenv
    load_dotenv()
    import logging
    logging.basicConfig(level=logging.INFO)