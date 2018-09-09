import json
from pymongo import MongoClient

# DB details
client = MongoClient("mongodb://admin:password123@ds149252.mlab.com:49252/fbhackathon")
db = client['fbhackathon']

# JSON to store
page = open("logs/log.json", 'r')
parsed = json.loads(page.read())
print(parsed)
db.test.insert(parsed)
