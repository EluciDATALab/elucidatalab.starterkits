"""used modules from elucidata.tools are stored statically on starter_kits package;
this script recursively replaces the import occurrences of 'elucidata.tools' with
'starter_kits'"""

from pathlib import Path
import numpy as np


def replace_in_line(line):
    if 'elucidata.tools' in line:
        replaced = 1
        line = line.replace('elucidata.tools', 'starterkits')
    else:
        replaced = 0

    return replaced, line


def replace_in_file(fname):
    with open(fname, 'r') as file:
        lines = file.readlines()

    replaced = 0
    newlines = []
    for l in lines:
        r, l = replace_in_line(l)
        newlines.append(l)
        replaced += r
    replaced = np.sum(replaced)
    print(f'{fname}: Replaced {replaced} occurrences of elucidata.tools')
    newlines = "".join(newlines)
    with open(fname, "w") as file:
        file.write(newlines)


if __name__ == "__main__":
    files = list(Path('.').rglob('*py'))

    [replace_in_file(f)
     for f in files
     if str(f) != 'replace_elucidata_tools_imports.py']
