"""
Outputs diagnostic information to help with debugging SLEAP installation.
"""

print_to_screen = False
output_text = ""


def print_(text=""):
    global output_text
    if print_to_screen:
        print(f"{text}")
    else:
        output_text += f"{text}\n"


def header(text):
    print_(f"\n=========={text.upper()}==========\n")


def label(text, obj):
    print_(f"{text}:\t\t\t{obj}")


def call(command):
    import subprocess

    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True
        )
        print_(result.stdout.decode())
    except FileNotFoundError:
        print_(f"unable to locate {command[0]}")
    except subprocess.CalledProcessError as e:
        print_(f"call to {command[0]} failed")
        print_(e.output.decode())


def call_self(call_args):
    import sys

    python_path = sys.executable
    self_path = sys.argv[0]
    call([python_path, self_path] + call_args)


def system_section():
    header("SYSTEM")

    import datetime
    import platform
    import os

    label("utc", datetime.datetime.utcnow())
    label("python", platform.python_version())
    system_info = (
        f"{platform.system()}, {platform.machine()}, "
        f"{platform.release()}, {platform.version()}"
    )
    label("system", system_info)
    label("path", os.getenv("PATH"))


def imports_section():
    header("IMPORTS")

    try:
        import sleap

        label("sleap import", True)
        label("sleap path", sleap.__file__)

        sleap_version = "not found"
        if hasattr(sleap, "__version__"):
            sleap_version = sleap.__version__
        label("sleap version", sleap_version)
    except Exception:
        label("sleap import", False)

    try:
        import PySide2

        label("pyside2 import", True)
        label("pyside path", PySide2.__file__)

        call_self(["--gui-check"])

    except Exception:
        label("pyside2 import", False)

    try:
        import PySide6

        label("pyside6 import", True)
        label("pyside path", PySide6.__file__)

        call_self(["--gui-check"])

    except Exception:
        label("pyside6 import", False)

    try:
        label("cv2 import", True)
    except Exception:
        label("cv2 import", False)


def tensorflow_section():
    header("TENSORFLOW")
    label("tensorflow import", False)
    label("tensorflow", "Neural network functionality has been removed")


def package_section():
    header("CONDA")
    call(["conda", "list"])

    header("PIP")
    call(["pip", "freeze"])


def nvidia_section():
    header("NVIDIA")
    call(["nvidia-smi"])


def git_section():
    header("GIT")
    call(["git", "rev-parse", "HEAD"])
    call(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def get_diagnostics(output_path=None):
    system_section()
    imports_section()
    git_section()
    tensorflow_section()
    package_section()
    nvidia_section()

    if output_path:
        with open(output_path, "w") as f:
            f.write(output_text)
    else:
        print(output_text)


def gui_check():
    from qtpy.QtWidgets import QApplication

    QApplication([])
    print("successfully created QApplication instance")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", help="Path for saving output", type=str, default=None
    )
    parser.add_argument(
        "--gui-check",
        help="Check if Qt GUI widgets can be used",
        action="store_const",
        const=True,
        default=False,
    )

    args = parser.parse_args()

    if args.gui_check:
        gui_check()
    else:
        get_diagnostics(output_path=args.output)


if __name__ == "__main__":
    main()
