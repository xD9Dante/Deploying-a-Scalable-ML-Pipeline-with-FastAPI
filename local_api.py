import json

import requests

url = "http://127.0.0.1:8000/"
r = requests.get(url)

print(r.status_code)
print(r.json())



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO: send a POST using the data above
r = requests.post(url + "data/", data=json.dumps(data))

print(r.status_code)
print(r.json())
