import sys
import util
import toml

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

    with open(util.scanner_config_path(), 'w') as f:
        f.write(toml.dumps(results))

if __name__ == '__main__':
    main()
