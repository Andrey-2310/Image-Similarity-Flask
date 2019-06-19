from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from image_similarity.opencv_template import find_closest_images

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql:'
# db = SQLAlchemy(app)


if __name__ == '__main__':
    app.run()


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    image1_bytes = request.files['images'].read()
    detector = request.form["detector"]
    weight = float(request.form["weight"])

    similarities = find_closest_images(detector, weight, image1_bytes)

    return render_template('hello.html', image1 = request.files['images'],
                           similarities=similarities, weight = weight, detector = detector)


@app.route('/check_similarity', methods=['GET'])
def hello_man():
    return render_template('hello.html')
