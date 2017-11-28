import subprocess
try:
    from flask import Flask, request, send_from_directory
except ImportError:
    print('This example needs Flask to run. Try running:\n'
          'pip install flask')

app = Flask(__name__)


STATIC_DIR = 'examples/reverse_image_search/static'

# TODO(wcrichto): figure out how to prevent image caching

@app.route('/mystatic/<path:path>')
def mystatic(path):
    return send_from_directory('static', path)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        f.save('{}/query.jpg'.format(STATIC_DIR))
        subprocess.check_call(['python', 'examples/reverse_image_search/search.py'])
        return """
<img src="/mystatic/result0.jpg" />
<img src="/mystatic/result1.jpg" />
<img src="/mystatic/result2.jpg" />
<img src="/mystatic/result3.jpg" />
<img src="/mystatic/result4.jpg" />
"""
    else:
        return """
<form method="post" enctype="multipart/form-data">
    <input type="file" name="file">
    <input type="submit" value="Upload">
</form>
"""


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
