# print_color.py

class PrintColor:
    """Class for printing colored text in the terminal."""

    def _print_color(self, text: str, color_code: str):
        """Internal method to print colored text using ANSI escape codes."""
        print(f"\033[{color_code}m{text}\033[00m")

    def red(self, text: str) -> None:
        self._print_color(text, "91")

    def green(self, text: str) -> None:
        self._print_color(text, "92")

    def yellow(self, text: str) -> None:
        self._print_color(text, "93")

    def magenta(self, text: str) -> None:
        self._print_color(text, "95")

    def purple(self, text: str) -> None:
        self._print_color(text, "35")

    def cyan(self, text: str) -> None:
        self._print_color(text, "96")

    def light_gray(self, text: str) -> None:
        self._print_color(text, "37")

    def black(self, text: str) -> None:
        self._print_color(text, "30")


# สร้าง instance ของ PrintColor ชื่อ pr
pr = PrintColor()

# Test the main function
if __name__ == "__main__":
    pr.red("This is a red text.")
    pr.green("This is a green text.")
    pr.yellow("This is a yellow text.")
    pr.magenta("This is a magenta text.")
    pr.purple("This is a purple text.")
    pr.cyan("This is a cyan text.")
    pr.light_gray("This is a light gray text.")
    pr.black("This is a black text.")
