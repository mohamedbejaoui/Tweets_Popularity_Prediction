#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import json
import base64

# Access token : 1222264597547233280-mVEGi2OkWt1YZHSm22QCkSfcxO50Rj
# Access token secret : jbbRh7KHT53bLSXIaanm2DfNqMGxIUtu7dKt2gtfxViPb


# OAuth Credentials
TWITTER_OAUTH = {"API key": "Byw1V0k4xe8H29iZgGUHhOTRg",
                 "API secret key": "U84CBUvfXAqFOsBVZTH6HYd5hqaOIFAhWFmFVs3alsrxeHbqFu",
                 "Access token": "1222264597547233280-mVEGi2OkWt1YZHSm22QCkSfcxO50Rj"}

key_secret = '{}:{}'.format(TWITTER_OAUTH["API key"], TWITTER_OAUTH["API secret key"]).encode('ascii')
b64_encoded_key = base64.b64encode(key_secret).decode('ascii')

# AUTHENTICATION
auth_resp = requests.post("https://api.twitter.com/oauth2/token",
                          headers = {'Authorization': 'Basic {}'.format(b64_encoded_key),
                                     'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'},
                          data = {'grant_type': 'client_credentials'})
# Access token for the GET requests
access_token = auth_resp.json()['access_token'] 


# SEARCH OF RETWEET CASCADES
tweet_id = str(1221816925677936640)

search_resp = requests.get("https://api.twitter.com/1.1/statuses/retweets/"+tweet_id+".json",
                           headers = {'Authorization': 'Bearer {}'.format(access_token)})

search_data = json.loads(search_resp.content)

# Timestamp of Retweet
search_data[17]["created_at"]
# User who retweeted
search_data[17]["user"]["followers_count"]
# Number of Retweets of original Tweet
search_data[17]["retweet_count"]





# CHECK API's LIMITATIONS
limit_resp = requests.get("https://api.twitter.com/1.1/application/rate_limit_status.json",
                          headers = {'Authorization': 'Bearer {}'.format(access_token)})

limit_data = json.loads(limit_resp.content)
for topic in limit_data['resources']:
    for url in limit_data['resources'][topic]:
        if limit_data['resources'][topic][url]['remaining'] < limit_data['resources'][topic][url]['limit']:
            print(topic, '\n', url, '\n', limit_data['resources'][topic][url])


###