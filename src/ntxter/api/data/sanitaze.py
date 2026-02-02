BASE_CHAR_MAP = {
    "á": "a",
    "é": "e",
    "í": "i",
    "ó": "o",
    "ú": "u",
    "ñ": "n",
    "ü": "u",
    "Á": "A",
    "É": "E",
    "Í": "I",
    "Ó": "O",
    "Ú": "U",
    "Ñ": "N",
    "Ü": "U",
    "ä": "a",
    "ë": "e",
    "ï": "i",
    "ö": "o",
    "Ä": "A",
    "Ë": "E",
    "Ï": "I",
    "Ö": "O",
    "ß": "ss",
    "ç": "c",
    "Ç": "C",
    "à": "a",
    "è": "e",
    "ã": "a",
    "õ": "o",
    "À": "A",
    "È": "E",
    "Ã": "A",
    "Õ": "O",
    "â": "a",
    "ê": "e",
    "ô": "o",
    "Â": "A",
    "Ê": "E",
    "Ô": "O",
    "ù": "u",
    "Ù": "U",
    "º": "o",
    "ª": "a",
    r'~': '',
    r'‐': '-',
    r'–': '-',
    r'—': '-',
    r'“': '',
    r'”': '',
    r'‘': '',
    r'’': '',
    r"'": '',
    r'"': '',
    '.': '_',
    ',': '_',
    ';': '_',
    ':': '_',
    '?': '',
    '!': '',
    "^": "",
    "~": ""
}

BASE_STR_MAP = {
    "*": "_prod_",
    "+": "_plus_",
    "=": "_eq_",
    "<": "_lt_", #less than
    ">": "_gt_", #greater than
    "≥": "_geq_", #greater equal than
    "≤": "_leq_", #less equal than
    "/": "_div_",
    r'\&amp': '_and_',
    "&": "_and_",
    "%": "perc_",
    "$": "peso_",
    "€": "euro_",
    "#": "num_",
}

BASE_REGEX_MAP = {
    r"\([^)]*\)|\{[^}]*\}|\[[^\]]*\]": "", # remove text within parentheses, brackets, or braces
    r"_+": "_", # multiple underscores to single
    r"^_+|_+$": "", # leading or trailing underscores
    r"_+[dD][eE]_+": "_", # remove standalone 'de' with surrounding spaces
    r"_+[pP][oO][rR]_+": "_" # remove standalone 'por' with surrounding spaces
}

from ntxter.adapters.data.colnames_sanitizer import PandasFrameColnamesSanitizer