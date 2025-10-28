# Project description:
PROJECT_DESC = {
    'name': 'Value Creation of Data Platforms through Secondary Use',
    'version': 'Master Thesis',
    'author': 'Felix Haas',
    'description': 'This project explores the value creation of data platforms through secondary use, focusing on the integration and analysis of clinical trial data.',
    'license': 'MIT'
}

# Database configuration
MY_DB_CONFIG = {
    'dbname': 'MASTER',
    'user': 'postgres',
    'password': 'Nwa471995',
    'host': 'localhost',
    'port': 5432
}

AACT_DB_CONFIG = {
    'dbname': 'aact',
    'user': 'haasenstall',
    'password': 'Nwa471995',
    'host': 'aact-db.ctti-clinicaltrials.org',
    'port': 5432
}

# Data paths
DATA_PATHS = {
    "logger": {
        "main": "./logs/main.log",
        "testing": "./logs/testing_log.log",
        "scraper": "./logs/scraper_log.log",
        "db": "./logs/db_log.log",
        "api": "./logs/api_log.log",
        "descriptive": "./logs/descriptive_log.log"
    },
    "img": {
        "default": "./images/",
        "investigator": "./images/investigators/",
        "institution": "./images/institutions/",
        "trials": "./images/trials/",
        "requests": "./images/requests/",
        "platform": "./images/platforms/",
        "funk": {
            "calculations": "./images/funk/calculations/",
            "numbers": "./images/funk/numbers/"
        },
        "publications": "./images/publications/",
        "regression": "./images/regression/"
    },
    "data_file": "./data/",
    "results": "./data/results/",
    "descriptive": "./data/descriptive/"
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        }
    },
    'handlers': {
        'file_handler': {
            'class': 'logging.FileHandler',
            'filename': DATA_PATHS['logger']['main'],
            'formatter': 'simple',
            'level': 'WARNING',
            'mode': 'w'
        },
        'testing_handler': {
            'class': 'logging.FileHandler',
            'filename': DATA_PATHS['logger']['testing'],
            'formatter': 'detailed',
            'level': 'DEBUG',
            'mode': 'w'
        },
        'scraper_handler': {
            'class': 'logging.FileHandler',
            'filename': DATA_PATHS['logger']['scraper'],
            'formatter': 'simple',
            'level': 'WARNING',
            'mode': 'w'
        },
        'db_handler': {
            'class': 'logging.FileHandler',
            'filename': DATA_PATHS['logger']['db'],
            'formatter': 'simple',
            'level': 'ERROR',
            'mode': 'w'
        },
        'api_handler': {
            'class': 'logging.FileHandler',
            'filename': DATA_PATHS['logger']['api'],
            'formatter': 'simple',
            'level': 'ERROR',
            'mode': 'w'
        },
        'descriptive_handler': {
            'class': 'logging.FileHandler',
            'filename': DATA_PATHS['logger']['descriptive'],
            'formatter': 'detailed',
            'level': 'ERROR',
            'mode': 'w'
        }
    },
    'loggers': {
        'logger': {
            'handlers': ['file_handler'],
            'level': 'ERROR',
            'propagate': False
        },
        'testing_logger': {
            'handlers': ['testing_handler'],
            'level': 'DEBUG',
            'propagate': False
        },
        'scraper_logger': {
            'handlers': ['scraper_handler'],
            'level': 'WARNING',
            'propagate': False
        },
        'db_logger': {
            'handlers': ['db_handler'],
            'level': 'WARNING',
            'propagate': False
        },
        'api_logger': {
            'handlers': ['api_handler'],
            'level': 'ERROR',
            'propagate': False
        },
        'descriptive_logger': {
            'handlers': ['descriptive_handler'],
            'level': 'DEBUG',
            'propagate': False
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['file_handler'],
        'encoding': 'utf-8'
    }
}

# Project settings
PROJECT_SETTINGS = {
    "debug": True,
    "dry_run": False,
    "log_errors": True,
    "max_retries": 3,
    "retry_delay": 2,
    "timeout": 30,
    "log_level": "ERROR",
    "max_workers": 3,
    "omp_num_threads": 4,
    
    }

# === Additional configurations ===

REQUEST_SETTINGS = {
    "headers": {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"  # Updated Chrome version
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "document",  # Added security headers
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1"
    },
    "vivli": {
        "request_url": "https://vivli.org/approved-research-proposals/",
        "public_disclosure_url": "https://vivli.org/resources/public-disclosures/"
    },
    "csdr": {
        "request_url": "https://www.clinicalstudydatarequest.com/Metrics/Agreed-Proposals.aspx",
        "base_url": "https://www.clinicalstudydatarequest.com/",
        "public_disclosure_url": "https://www.clinicalstudydatarequest.com/Metrics/Published-Proposals.aspx"
    },
    "yoda": {
        "request_url": "https://yoda.yale.edu/metrics/submitted-requests-to-use-johnson-johnson-data/data-requests-johnson-and-johnson/?_paged={page}",
        "trial_url": "https://yoda.yale.edu/trials-search/?_paged={page}"
    },
    "crossref": {
        "base_url": "https://api.openalex.org/works/https://doi.org/{doi}",
        "pubmed_url": "https://api.openalex.org/works/pmid:{pmid}",
        "cited_by_url": "https://api.openalex.org/works?page={page}&filter=cites:{openalex_id}&sort=cited_by_count:desc&per_page={per_page}"
    }
}

# Platform Settings 
PLATTFORM_IDS = {
    "vivli": 1,
    "csdr": 2,
    "yoda": 3
}

NCT_ID_APIS = {
    "vivli": {
        "url": "https://vivli-prod-cus-srch.search.windows.net/indexes/studies/docs",
        "params": {
            "api-key": "C8237BFE70B9CC48489DC7DD84D88379",
            "api-version": "2016-09-01",
            "$top": 15, # changeable to max per page - unknown max
            "$skip": "{skip}",
            "search": "*",
            "$filter": "assignedAppType eq 'Default'",
            "$count": "true",
            "facet": [
                "studyDesign",
                "locationsOfStudySites,count:300,sort:value",
                "sponsorType",
                "contributorType", 
                "sponsorName,count:500,sort:value",
                "studyType",
                "actualEnrollment,interval:100",
                "funder,count:500",
                "orgName,count:500"
            ]
        }
    },
    "csdr": {
        "url": "",  
        "params": {}
    },
    "yoda": {
        "url": "",  
        "params": {}
    }
}
