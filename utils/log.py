import logging


def get_log(file_name, log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) 

    fh = logging.FileHandler(file_name, mode='a') 
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s]  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter) 
    fh.setFormatter(formatter)
    logger.addHandler(fh) 
    logger.addHandler(ch)
    return logger

