class TerminalColors:
    """
    Centralized ANSI color codes for prominent terminal logging.
    """
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

    @staticmethod
    def color_text(text: str, color: str) -> str:
        return f"{color}{text}{TerminalColors.RESET}"
