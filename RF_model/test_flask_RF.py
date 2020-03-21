#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

url = 'http://0.0.0.0:5000/api/'

data = [[0.7,0.22,36,0.8]]
j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
r_json = r.json()

print("r_json", float(r_json[1:-1]))

#