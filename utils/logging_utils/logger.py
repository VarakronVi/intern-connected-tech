# logger.py
"""
This module contains the configuration for logging in the project.
Setup:
    config.yaml is in the project root
    LOG_DEBUG: True
        1. Have the config.yaml file -> LOG_DEBUG: True -> to log the messages.
        2. Have the config.yaml file -> LOG_DEBUG: False -> to not log
        3. Have the config.yaml file -> LOG_DEBUG: None -> to not log
        4. If the config.yaml file is not present -> to log the messages.
Example:
    To use this module, import it and use the logger object to log messages.
        ''' ---------- Import the required libraries ---------- '''
        # Import custom modules
        from utils.helpers import logger

        logger.info("This is an information message.")
        logger.error("This is an error message.")
        logger.warning("This is a warning message.")
"""
# Setup the environment
import logging
import os
import time

# Create log file
date_today = time.strftime("%Y-%m-%d")
current_time = time.strftime("%Y-%m-%d-%H")

output_dir = f"logs/{date_today}"
os.makedirs(output_dir, exist_ok=True)

log_filename = f'{output_dir}/shared_log_{current_time}.log'

# Configure logging
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create logger
logger = logging.getLogger()


if __name__ == "__main__":
    logger.info("This is an information message.")
    logger.error("This is an error message.")
    logger.warning("This is a warning message.")
    logger.critical("This is a critical message.")