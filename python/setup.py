import sys
import toml
from scanner import ScannerConfig

def ask_questions(questions):
    results = {}
    for (key, question) in questions:
        if isinstance(question, list):
            results[key] = ask_questions(question)
        else:
            parts = key.split('#')
            should_ask = True
            if len(parts) > 1:
                key = parts[1]
                cond = parts[0].split(':')
                should_ask = results[cond[0]] == cond[1]

            if should_ask:
                while True:
                    sys.stdout.write(question + ': ')
                    val = sys.stdin.readline().strip()
                    if val == '': continue
                    results[key] = val
                    break
    return results

def main():
    questions = [
        ('scanner_path', 'Absolute path to Scanner directory'),
        ('storage', [
            ('type', 'Storage backend type (posix|gcs)'),
            ('db_path', 'Absolute path to Scanner database directory'),
            ('type:gcs#key_path', 'Absolute path to GCS JSON keyfile'),
            ('type:gcs#cert_path', 'Absolute path to GCS SSL certificate'),
            ('type:gcs#bucket', 'GCS bucket to use')
        ])
    ]

    results = ask_questions(questions)

    path = ScannerConfig.default_config_path()
    with open(path, 'w') as f:
        f.write(toml.dumps(results))

if __name__ == '__main__':
    main()
