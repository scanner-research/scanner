from setuptools import setup, find_packages, Extension
import os
import os.path
import shutil
import glob
from sys import platform

SCRIPT_DIR = '.'
PYTHON_DIR = os.path.join(SCRIPT_DIR, 'python')
SCANNERPY_DIR = os.path.join(SCRIPT_DIR, 'python', 'scannerpy')
SCANNER_DIR = os.path.join(SCRIPT_DIR, '.')
ROOT_DIR = SCANNER_DIR
BUILD_DIR = os.path.join(SCANNER_DIR, 'build')
PIP_DIR = os.path.join(BUILD_DIR, 'pip')

def main():
    # Make a pip directory in the build directory
    shutil.rmtree(PIP_DIR, ignore_errors=True)
    shutil.copytree(PYTHON_DIR, PIP_DIR)
    #os.makedirs(PIP_DIR, exist_ok=True)
    #os.makedirs(PIP_DIR + '/scanner', exist_ok=True)

    # Copy python into pip directory
    #shutil.copytree(SCANNERPY_DIR, PIP_DIR + '/scannerpy')

    if platform == 'linux' or platform == 'linux2':
        EXT = '.so'
    else:
        EXT = '.dylib'

    # Copy libraries into pip directory
    os.makedirs(os.path.join(PIP_DIR, 'scannerpy', 'lib'))
    library = os.path.join(BUILD_DIR, 'libscanner' + EXT)
    name = os.path.splitext(os.path.basename(library))[0]
    shutil.copyfile(library, os.path.join(PIP_DIR, 'scannerpy', 'lib', name + EXT))


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
        os.path.join(PIP_DIR, 'scannerpy', 'include', 'scanner'), '*.hpp')
    copy_partial_tree(
        os.path.join(SCANNER_DIR, 'scanner'),
        os.path.join(PIP_DIR, 'scannerpy', 'include', 'scanner'), '*.inl')

    copy_partial_tree(
        os.path.join(BUILD_DIR, 'scanner'),
        os.path.join(PIP_DIR, 'scannerpy', 'include', 'scanner'), '*.h')
    copy_partial_tree(
        os.path.join(BUILD_DIR, 'scanner'),
        os.path.join(PIP_DIR, 'scannerpy', 'include', 'scanner'), '*.hpp')

    include_files = glob_files(
        os.path.join(PIP_DIR, 'scannerpy', 'include'), 'include')

    package_data = {
        'scannerpy': ['lib/*.so', 'lib/*' + EXT] + include_files + cmake_files
       }

    REQUIRED_PACKAGES = [
        'protobuf == 3.6.1', 'grpcio == 1.16.0', 'toml >= 0.9.2',
        'numpy >= 1.12.0,<=1.16.0', 'tqdm >= 4.19.5', 'cloudpickle >=0.5.3,<=0.6.1',
        'attrs == 18.2.0', 'psutil == 5.6.1'
       ]

    TEST_PACKAGES = ['pytest']

    if platform == 'linux' or platform == 'linux2':
        REQUIRED_PACKAGES.append('python-prctl >= 1.7.0')

    # Borrowed from https://github.com/pytorch/pytorch/blob/master/setup.py
    def make_relative_rpath(path):
        if platform == 'linux' or platform == 'linux2':
            return '-Wl,-rpath,$ORIGIN/' + path
        else:
            return '-Wl,-rpath,@loader_path/' + path


    module1 = Extension(
        'scannerpy._python',
        include_dirs = [ROOT_DIR,
                        os.path.join(ROOT_DIR, 'build'),
                        os.path.join(ROOT_DIR, 'thirdparty', 'install', 'include')],
        libraries = ['scanner'],
        library_dirs = [ROOT_DIR,
                        os.path.join(ROOT_DIR, 'build'),
                        os.path.join(ROOT_DIR, 'thirdparty', 'install', 'lib')],
        sources = [os.path.join(ROOT_DIR, 'scanner/engine/python.cpp')],
        extra_compile_args=['-std=c++11'],
        extra_link_args=[make_relative_rpath('lib')])

    setup(
        name='scannerpy',
        version='0.2.22',
        description='Efficient video analysis at scale',
        long_description='',
        url='https://github.com/scanner-research/scanner',
        author='Alex Poms and Will Crichton',
        author_email='wcrichto@cs.stanford.edu',
        package_dir={'': PIP_DIR},
        packages=find_packages(where=PIP_DIR),
        install_requires=REQUIRED_PACKAGES,
        setup_requires=['pytest-runner'],
        tests_require=TEST_PACKAGES,
        include_package_data=True,
        package_data=package_data,
        zip_safe=False,
        license='Apache 2.0',
        keywords='video distributed gpu',
        ext_modules=[module1],
       )

if __name__ == "__main__":
    main()
