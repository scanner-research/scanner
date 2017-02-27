from setuptools import setup, find_packages
import os

REQUIRED_PACKAGES = [
    'protobuf >= 3.1.0',
    'grpcio >= 1.1.0',
    'toml >= 0.9.2',
    'enum34 >= 1.1.6',
    'numpy >= 1.12.0',
    'scipy >= 0.18.1',
    'storehouse >= 0.1.0'
]

package_data = {
    'scannerpy': [
        'build/*.so',
    ]
}

os.system('mkdir -p python/scannerpy/include && '
          'ln -fs ../../../scanner python/scannerpy/include &&'
          'ln -fs ../../build python/scannerpy')

def get_build_dirs(d):
    return [t[0]+'/*' for t in os.walk('build/'+d) if 'CMakeFiles' not in t[0]]

package_data['scannerpy'] += get_build_dirs('scanner')
package_data['scannerpy'] += get_build_dirs('stdlib')
package_data['scannerpy'] += ['include/{}/*.h'.format(t[0])
                              for t in os.walk('scanner')]

setup(
    name='scannerpy',
    version='0.1.7',
    description='Efficient video analysis at scale',
    long_description='',
    url='https://github.com/scanner-research/scanner',
    author='Alex Poms and Will Crichton',
    author_email='wcrichto@cs.stanford.edu',

    package_dir={'': 'python'},
    packages=find_packages(where='python'),
    install_requires=REQUIRED_PACKAGES,
    package_data=package_data,

    license='Apache 2.0',
    keywords='video distributed gpu',
)
