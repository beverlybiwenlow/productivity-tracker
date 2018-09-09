from flask import Flask
import json
from flask_pymongo import PyMongo

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://admin:password123@ds149252.mlab.com:49252/fbhackathon"
mongo = PyMongo(app)

myCursor = mongo.db.test.find()
entries = myCursor[:]
data_list = entries[0]
print(data_list)

@app.route('/')
def return_value1():
    return "Success. Server works."

@app.route('/activity-category-pie-chart')
def return_value2():

    category_obj = {}

    for timestamp in data_list:
        if timestamp != "_id":
            category = data_list[timestamp]["Category"]
            if category in category_obj:
                print("YES")
                print(category)
                category_obj[category] += 1
            else:
                print("NO")
                print(category)
                category_obj[category] = 1
    print(category_obj)
    return json.dumps(category_obj)

@app.route('/keystrokes-against-time')
def return_value3():
    keystrokes_obj = {}
    keystrokes_array = []
    timestamp_array = []
    for timestamp in data_list:
        if timestamp != "_id":
            timestamp_array.append(timestamp)
            keystrokes_array.append(data_list[timestamp]["Keystrokes"])
    keystrokes_obj["Timestamps"] = timestamp_array
    keystrokes_obj["Keystrokes"] = keystrokes_array
    return json.dumps(keystrokes_obj)


if __name__ == "__main__":
    app.run(debug=True)