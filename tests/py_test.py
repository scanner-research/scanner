from scannerpy import Database, Config
import tempfile
import toml

def test_new_database():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        cfg = Config.default_config()
        cfg['storage']['db_path'] = tempfile.mkdtemp()
        f.write(toml.dumps(cfg))
        path = f.name

    db = Database(path)
