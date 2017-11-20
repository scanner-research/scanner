from .. import Config
import subprocess as sp

c = Config()

build_flags_output = [
    s.strip()
    for s in sp.check_output(
        [c.config['scanner_path'] + '/build/scanner/engine/build_flags']).split("\n")
]
include_dirs = build_flags_output[0].split(";")
include_dirs.append(c.module_dir + "/include")
include_dirs.append(c.module_dir + "/build")
flags = '{include} -std=c++11 -fPIC -shared -L{libdir} -lscanner {other}'
print(flags.format(
    include=" ".join(["-I " + d for d in include_dirs]),
    libdir='{}/build'.format(c.module_dir),
    other=build_flags_output[1]))
