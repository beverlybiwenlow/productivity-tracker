import json
from pymongo import MongoClient

client = MongoClient("mongodb://admin:password123@ds149252.mlab.com:49252/fbhackathon")

# Get the sampleDB database
db = client['fbhackathon']


db.test.insert({"testval": {"first": 1, "second":2}})
# db.close()