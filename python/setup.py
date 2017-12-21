from setuptools import setup, find_packages
import os
import os.path
import shutil

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCANNERPY_DIR = os.path.join(SCRIPT_DIR, 'scannerpy')
SCANNER_DIR = '.'
BUILD_DIR = os.path.join(SCANNER_DIR, 'build')
PIP_DIR = os.path.join(BUILD_DIR, 'pip')

# Make a pip directory in the build directory
shutil.rmtree(PIP_DIR, ignore_errors=True)
#os.makedirs(PIP_DIR)
#os.makedirs(PIP_DIR + '/scanner')
#os.makedirs(PIP_DIR + '/scanner/stdlib')

# Copy python into pip directory
shutil.copytree(SCRIPT_DIR, PIP_DIR)
#shutil.copytree(SCANNERPY_DIR, PIP_DIR + '/scannerpy')

# Copy libraries into pip directory
LIBRARIES = [
    os.path.join(BUILD_DIR, 'libscanner.so'),
    os.path.join(BUILD_DIR, 'stdlib', 'libstdlib.so')
]
for library in LIBRARIES:
    shutil.copy(library, PIP_DIR + '/scannerpy/')

PROTO_HEADERS = [
    (os.path.join(BUILD_DIR, 'scanner', 'metadata_pb2.py'),
     os.path.join(PIP_DIR, 'scanner')),

    (os.path.join(BUILD_DIR, 'scanner', 'types_pb2.py'),
     os.path.join(PIP_DIR, 'scanner')),

    (os.path.join(BUILD_DIR, 'scanner', 'engine', 'rpc_pb2.py'),
     os.path.join(PIP_DIR, 'scanner', 'engine')),

    (os.path.join(BUILD_DIR, 'scanner', 'engine', 'rpc_pb2_grpc.py'),
     os.path.join(PIP_DIR, 'scanner', 'engine')),

    (os.path.join(BUILD_DIR, 'scanner', 'types_pb2.py'),
     os.path.join(PIP_DIR, 'scanner')),

    (os.path.join(BUILD_DIR, 'stdlib', 'stdlib_pb2.py'),
     os.path.join(PIP_DIR, 'scanner', 'stdlib'))
]
for src, dest in PROTO_HEADERS:
    shutil.copy(src, dest)

package_data = {
    'scannerpy': [
        './*.so'
    ]
}

REQUIRED_PACKAGES = [
    'protobuf == 3.4.0',
    'grpcio == 1.7.3',
    'toml >= 0.9.2',
    'enum34 >= 1.1.6',
    'numpy >= 1.12.0',
    'scipy >= 0.18.1',
    'storehouse >= 0.1.0'
]

print(find_packages(where=PIP_DIR))
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
