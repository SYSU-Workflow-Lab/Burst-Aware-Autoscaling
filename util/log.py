import logging
import os

OUTPUT_DIR = os.path.join('.', 'output')
LOG_DIR = os.path.join(OUTPUT_DIR,'log')

def logger_register(logger_name: str = '', outputFilename: str = "main_informer_Function.log") -> None:
    """
        设置日志格式，使其同时在文件与命令行中进行输出
        该函数需要被放在最前，先于第一个执行的logging函数
    :param outputFilename: 日志输出的文件名称
    :param logger_name: str 日志记录器的名称
    :return: None
    """
    if not os.path.exists(OUTPUT_DIR):
       os.mkdir(OUTPUT_DIR)
    if not os.path.exists(LOG_DIR):
       os.mkdir(LOG_DIR)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s  %(filename)s:%(lineno)d : %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %A %H:%M:%S'
    )

    fileHandler = logging.FileHandler(os.path.join('output', 'log', outputFilename))  # 默认mode=a
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)