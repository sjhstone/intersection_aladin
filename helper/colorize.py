import colorama
from problem import PRINT_LEVEL

def color_print(category, level, msg):
    """print colorized console output
    """
    if level <= PRINT_LEVEL:
        print(msg)
        """ unknown problem was caused by colorama
        colorama.init()
        if category == 'ok':
            print(colorama.Fore.GREEN  + '[   OK ]' + colorama.Style.RESET_ALL, msg)
        elif category == 'info':
            print(colorama.Fore.CYAN   + '[ INFO ]' + colorama.Style.RESET_ALL, msg)
        elif category == 'warning':
            print(colorama.Fore.YELLOW + '[ WARN ]' + colorama.Style.RESET_ALL, msg)
        elif category == 'error':
            print(colorama.Fore.RED    + '[  ERR ]' + colorama.Style.RESET_ALL, msg)
        elif category == 'debug':
            print(colorama.Fore.BLUE   + '_debug__' + colorama.Style.RESET_ALL, msg)
        """
    else:
        pass
