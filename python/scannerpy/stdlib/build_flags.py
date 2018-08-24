import os.path
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def get_include():
    return os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'include'))

def print_include():
    sys.stdout.write(get_include())

def get_lib():
    return os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'lib'))

def print_lib():
    sys.stdout.write(get_lib())

def get_cmake():
    return os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'cmake', 'Op.cmake'))

def print_cmake():
    sys.stdout.write(get_cmake())

def get_compile_flags():
    return (
        '-std=c++11 -I{include}'.format(
            include=get_include()))

def get_link_flags():
    return (
        '-L{libdir} -lscanner'.format(
            libdir=get_lib()))

def print_compile_flags():
    sys.stdout.write(get_compile_flags())

def print_link_flags():
    sys.stdout.write(get_link_flags())
