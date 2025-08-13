import logging

class LogDriver:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
    # 
    @classmethod
    def config(cls, path, project_name, level=logging.DEBUG):
        inst = cls(path)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(context)s | %(message)s', datefmt='%Y-%m-%d')
        file_handler = logging.FileHandler(inst.path, mode='w')
        file_handler.setFormatter(formatter)
        #
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        #
        logger = logging.getLogger(f"{project_name}")
        logger.setLevel(level)
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False
        #
        inst.log = logger
        return inst
    #
    def info(self, context: str, msg: str):
        self.log.info(msg, extra={"context": context})
    #
    def debug(self, context: str, msg: str):
        self.log.debug(msg, extra={"context": context})
    #
    def error(self, context: str, msg: str):
        self.log.debug(msg, extra={"context": context})