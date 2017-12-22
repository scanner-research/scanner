from __future__ import absolute_import, division, print_function, unicode_literals
import os.path

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def get_include():
    return os.path.join(SCRIPT_DIR, '..', 'include')

def get_lib():
    return os.path.join(SCRIPT_DIR, '..')

def get_cmake():
    return os.path.join(SCRIPT_DIR, '..', 'cmake')

def get_flags():
    return (
        '-std=c++11 -I{include} -L{libdir} -lscanner'.format(
            include=get_include(),
            libdir=get_lib()))

