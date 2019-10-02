
RED   = "\033[1;31m"
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[1;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"

formats = {'red': RED, 'blue': BLUE, 'cyan': CYAN, 'green': GREEN,
           'bold': BOLD, 'reverse': REVERSE}


def format_print(text, formatters):
    tmp_str = ''
    for f in formatters:
        tmp_str += str(formats[f])
    tmp_str += repr(text)
    tmp_str += RESET
    print(tmp_str)

def warning(text):
    format_print(text, ('bold', 'red'))
