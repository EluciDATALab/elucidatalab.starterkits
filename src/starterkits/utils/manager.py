# ©, 2019, Sirris
# owner: HCAB

"""
A collection of all-purpose functions.

Existing functions:
- try_package
- search_files
"""

import os
import re

from tqdm import tqdm
from ipypb import track
import pkg_resources


def try_package(x):
    """
    Test if installed package matches requirements and print warning if not

    :param x : str. Package name and version in format 'PKG==VERSION' ('<=' or
    '>=' comparisons are also accepted)

    """
    # get operator
    op = re.findall('[><=]+', x)[0]

    pkg, pkg_version = x.split(op)
    try:
        version = pkg_resources.get_distribution(pkg).version
        if eval("'%s'%s'%s'" % (version, op, pkg_version)):
            pass
        else:
            print('Warning: this notebook was tested with version %s of %s, '
                  'but you have %s installed' %
                  (pkg_version, pkg, version))
    except Exception as e:
        print(e)


def search_files(directory='.', str_match=''):
    """
    Recursively look for files in directory

    :param directory : str. directory to look for files in. Default : '.'
    :param str_match : str. regex to match files in directory to. Default : ''

    :returns directory of file

    """
    file_loc = []
    for dirpath, dirnames, files in os.walk(directory):
        for name in files:
            if re.match(str_match, name):
                file_loc.append(os.path.join(dirpath, name))
    return file_loc


def progressbar(x, verbal=True, desc='', total=0):
    """
    Wrapper around tqdm/ipypb.track progressbars

    :param x: iteration variable
    :param verbal: bool. whether to print output
    :param desc: str. Description of progress
    :param total: int. Total number of iterators
    :return:
    """
    if verbal:
        if total == 0:
            try:
                len(x)
                return track(x, label=desc, total=total)
            except Exception as e:
                print(e)
                return tqdm(x, desc=desc, total=total)
        else:
            return track(x, label=desc, total=total)
    else:
        return x
