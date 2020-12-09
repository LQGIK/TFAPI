from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import tf


# CONFIG
app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True
app.config["TEMPLATES_AUTO_RELOAD"] = True
CORS(app)



@app.before_request
def before():
    print("This is executed BEFORE each request.")


@app.route('/setModel/', methods=['POST'])
def setModel():
    '''
    Accepts POST req of layers {nodes, activation, etc} to create a tensorflow model that is saved
    '''

    data = request.get_json()
    model = tf.getModel(data)
    print(model.summary())
    print(data)
    return jsonify(data)



@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    return render_template("index.html") 


@app.route('/<int:number>/')
def incrementer(number):
    return "Incremented number is " + str(number+1)


@app.route('/numbers/')
def print_list():
    return jsonify(
        list(range(5))
    )


@app.route('/<string:name>/')
def hello(name):
    return jsonify({
        'name': name,
        'address':'India'
    })


if __name__ == '__main__':
    app.run(host='localhost', port=8080)