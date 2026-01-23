import logging

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s"

    )
    
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
