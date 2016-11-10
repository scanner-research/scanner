import sys
import toml
from scanner import ScannerConfig

def main():
    questions = {
        'scanner_path': 'Absolute path to Scanner directory',
        'db_path': 'Absolute path to Scanner database directory'
    }

    results = {}
    for key, question in questions.iteritems():
        sys.stdout.write(question + ': ')
        val = sys.stdin.readline().strip()
        results[key] = val

    path = ScannerConfig.default_config_path()
    with open(path, 'w') as f:
        f.write(toml.dumps(results))

if __name__ == '__main__':
    main()
