from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/image')
def get_image():
    return send_from_directory('images', 'test.png')

if __name__ == "__main__":
    app.run(debug=True)