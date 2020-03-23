import os

doctree = ""

for root, dirs, files in os.walk("../sleap"):
    # remove leading "../"
    root = root[3:]

    for file in sorted(files):
        if file.endswith(".py") and not file.startswith("_"):
            full = os.path.join(root, file)
            full = full[:-3].replace(os.sep, ".")
            doctree += f"   {full}\n"

# get the api doc header
with open("api_head.rst", "r") as f:
    api_head = f.read()

# write file for api doc with header + doctree
with open("api.rst", "w") as f:
    f.write(api_head)
    f.write(doctree)
