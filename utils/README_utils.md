project_root/
├── utils/
│   ├── io_utils/                # Input/Output handling utilities
│   │   ├── file_operations.py   # File operations (read/write, directory handling)
│   │   ├── config_handler.py    # Configuration file handler (JSON, YAML, etc.)
│   ├── data_utils/              # Data processing utilities
│   │   ├── data_cleaning.py     # Data cleaning (e.g., handling missing values)
│   │   ├── data_transformation.py # Data transformations (e.g., normalization)
│   ├── logging_utils/           # Logging and debugging utilities
│   │   ├── log_setup.py         # Logger setup and configurations
│   │   ├── log_formatter.py     # Custom log formatters
│   ├── network_utils/           # Network and API related utilities
│   │   ├── http_requests.py     # HTTP request handling (GET/POST)
│   │   ├── url_parser.py        # URL parsing and management
│   ├── string_utils/            # String manipulation utilities
│   │   ├── text_formatter.py    # String formatting (e.g., camelCase to snake_case)
│   │   ├── regex_utils.py       # Regular expressions utilities
│   ├── image_utils/             # Image processing utilities
│   │   ├── image_resizer.py     # Image resizing and format conversion
│   │   ├── image_filter.py      # Image filtering and transformations
│   └── __init__.py              # Make `utils` a package
