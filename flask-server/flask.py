from flask import Flask
import json
from flask_pymongo import PyMongo

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://admin:password123@ds149252.mlab.com:49252/fbhackathon"
mongo = PyMongo(app)

@app.route('/')
def hello():
    # just testing stuff
    myCursor = mongo.db.test.find()
    entries = myCursor[:]
    for value in entries:
        print(value)
    return "hello"

if __name__ == "__main__":
    app.run(debug=True)