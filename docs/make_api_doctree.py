import os


ignore_modules = [
    "sleap.gui",
    "sleap.diagnostic",
    # "sleap.message",
    "sleap.prefs",
    "sleap.rangelist",
    "sleap.version",
]

def make_api_doctree():
    doctree = ""

    for root, dirs, files in os.walk("../sleap"):
        # remove leading "../"
        root = root[3:]

        for file in sorted(files):
            if file.endswith(".py") and not file.startswith("_"):
                full = os.path.join(root, file)
                full = full[:-3].replace(os.sep, ".")

                ignore = False
                for ignore_module in ignore_modules:
                    if full.startswith(ignore_module):
                        ignore = True
                        break
                if not ignore:
                    doctree += f"   {full}\n"

    # get the api doc header
    with open("_templates/api_head.rst", "r") as f:
        api_head = f.read()

    # write file for api doc with header + doctree
    with open("api.rst", "w") as f:
        f.write("..\n  This file is auto-generated.\n\n")
        f.write(api_head)
        f.write(doctree)


if __name__ == "__main__":
    make_api_doctree()