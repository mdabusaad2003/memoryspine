import urllib.request, json, urllib.error

# Test 1: Raw endpoint
req1 = urllib.request.Request('http://127.0.0.1:8091/embedding', data=json.dumps({'content': 'test'}).encode(), headers={'Content-Type': 'application/json'})
try:
    res1 = urllib.request.urlopen(req1)
    print("Raw OK:", res1.read())
except urllib.error.HTTPError as e:
    print("Raw Error 500 body:", e.read())

# Test 2: OpenAI endpoint
req2 = urllib.request.Request('http://127.0.0.1:8091/v1/embeddings', data=json.dumps({'input': 'test'}).encode(), headers={'Content-Type': 'application/json'})
try:
    res2 = urllib.request.urlopen(req2)
    print("OpenAI OK:", res2.read()[:150], "...")
except urllib.error.HTTPError as e:
    print("OpenAI Error body:", e.read())
