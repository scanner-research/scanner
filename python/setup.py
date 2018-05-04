from setuptools import setup, find_packages
import os
import os.path
import shutil
import glob
from sys import platform

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCANNERPY_DIR = os.path.join(SCRIPT_DIR, 'scannerpy')
SCANNER_DIR = '.'
BUILD_DIR = os.path.join(SCANNER_DIR, 'build')
PIP_DIR = os.path.join(BUILD_DIR, 'pip')

# Make a pip directory in the build directory
shutil.rmtree(PIP_DIR, ignore_errors=True)
shutil.copytree(SCRIPT_DIR, PIP_DIR)
#os.makedirs(PIP_DIR)
#os.makedirs(PIP_DIR + '/scanner')
#os.makedirs(PIP_DIR + '/scanner/stdlib')

# Copy python into pip directory
#shutil.copytree(SCANNERPY_DIR, PIP_DIR + '/scannerpy')

if platform == 'linux' or platform == 'linux2':
    EXT = '.so'
else:
    EXT = '.dylib'

# Copy libraries into pip directory
LIBRARIES = [
    os.path.join(BUILD_DIR, 'libscanner' + EXT),
    os.path.join(BUILD_DIR, 'stdlib', 'libstdlib' + EXT)
]
for library in LIBRARIES:
    name = os.path.splitext(os.path.basename(library))[0]
    shutil.copyfile(library, os.path.join(PIP_DIR, 'scannerpy', name + '.so'))


def copy_partial_tree(from_dir, to_dir, pattern):
    dest_paths = []
    try:
        os.makedirs(to_dir)
    except:
        pass
    for f in glob.glob(os.path.join(from_dir, pattern)):
        print(f)
        shutil.copy(f, to_dir)
        dest_paths.append(os.path.join(to_dir, os.path.basename(f)))

    # List all directories in from_dir
    for d in [
            p for p in os.listdir(from_dir)
            if os.path.isdir(os.path.join(from_dir, p))
    ]:
        print('dir', d)
        dest_paths += copy_partial_tree(
            os.path.join(from_dir, d), os.path.join(to_dir, d), pattern)
    return dest_paths


def glob_files(path, prefix=''):
    all_paths = os.listdir(path)
    files = [
        os.path.join(prefix, p) for p in all_paths
        if os.path.isfile(os.path.join(path, p))
    ]
    for d in [p for p in all_paths if os.path.isdir(os.path.join(path, p))]:
        files += glob_files(
            os.path.join(path, d), prefix=os.path.join(prefix, d))
    return files


# Copy built protobuf python files
copy_partial_tree(
    os.path.join(BUILD_DIR, 'scanner'), os.path.join(PIP_DIR, 'scanner'),
    '*.py')
copy_partial_tree(
    os.path.join(BUILD_DIR, 'stdlib'),
    os.path.join(PIP_DIR, 'scanner', 'stdlib'), '*.py')

# Copy cmake files
os.makedirs(os.path.join(PIP_DIR, 'scannerpy', 'cmake'))
shutil.copy(
    os.path.join(SCANNER_DIR, 'cmake', 'Util', 'Op.cmake'),
    os.path.join(PIP_DIR, 'scannerpy', 'cmake'))
copy_partial_tree(
    os.path.join(SCANNER_DIR, 'cmake', 'Modules'),
    os.path.join(PIP_DIR, 'scannerpy', 'cmake', 'Modules'), '*')

cmake_files = glob_files(os.path.join(PIP_DIR, 'scannerpy', 'cmake'), 'cmake')

# Copy scanner headers
copy_partial_tree(
    os.path.join(SCANNER_DIR, 'scanner'),
    os.path.join(PIP_DIR, 'scannerpy', 'include', 'scanner'), '*.h')
copy_partial_tree(
    os.path.join(SCANNER_DIR, 'scanner'),
    os.path.join(PIP_DIR, 'scannerpy', 'include', 'scanner'), '*.inl')

copy_partial_tree(
    os.path.join(BUILD_DIR, 'scanner'),
    os.path.join(PIP_DIR, 'scannerpy', 'include', 'scanner'), '*.h')

include_files = glob_files(
    os.path.join(PIP_DIR, 'scannerpy', 'include'), 'include')

package_data = {
    'scannerpy': ['./*.so', './*' + EXT] + include_files + cmake_files
}

REQUIRED_PACKAGES = [
    'protobuf == 3.4.0', 'grpcio == 1.7.3', 'toml >= 0.9.2', 'enum34 >= 1.1.6',
    'numpy >= 1.12.0', 'scipy >= 0.18.1', 'tqdm >= 4.19.5',
    'cloudpickle >= 0.5.2'
]
if platform == 'linux' or platform == 'linux2':
    REQUIRED_PACKAGES.append('python-prctl >= 1.7.0')

setup(
    name='scannerpy',
    version='0.1.13',
    description='Efficient video analysis at scale',
    long_description='',
    url='https://github.com/scanner-research/scanner',
    author='Alex Poms and Will Crichton',
    author_email='wcrichto@cs.stanford.edu',
    package_dir={'': PIP_DIR},
    packages=find_packages(where=PIP_DIR),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    package_data=package_data,
    zip_safe=False,
    license='Apache 2.0',
    keywords='video distributed gpu',
)
