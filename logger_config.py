import logging
import logging.handlers
import sys
import os
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama for Windows color support
init()

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)

def setup_logging(app_name="MindfulOdyssey"):
    """Setup comprehensive logging system."""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Generate log filenames with timestamps
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_log = f'logs/mino_{timestamp}.log'
    error_log = f'logs/errors_{timestamp}.log'
    chat_log = f'logs/chat_{timestamp}.log'

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console handler with color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    colored_formatter = ColoredFormatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(colored_formatter)
    root_logger.addHandler(console_handler)

    # Main rotating file handler
    main_handler = logging.handlers.RotatingFileHandler(
        main_log,
        maxBytes=1024*1024,  # 1MB
        backupCount=5
    )
    main_handler.setLevel(logging.DEBUG)
    main_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    root_logger.addHandler(main_handler)

    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        error_log,
        maxBytes=1024*1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s\n'
        'Exception: %(exc_info)s'
    ))
    root_logger.addHandler(error_handler)

    # Chat history handler
    chat_handler = logging.handlers.RotatingFileHandler(
        chat_log,
        maxBytes=1024*1024,
        backupCount=3
    )
    chat_handler.setLevel(logging.INFO)
    chat_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s'
    ))

    # Create chat logger
    chat_logger = logging.getLogger('chat')
    chat_logger.addHandler(chat_handler)
    chat_logger.setLevel(logging.INFO)

    # Log startup information
    root_logger.info(f"{app_name} logging system initialized")
    root_logger.info(f"Main log: {main_log}")
    root_logger.info(f"Error log: {error_log}")
    root_logger.info(f"Chat log: {chat_log}")

    return root_logger, chat_logger
