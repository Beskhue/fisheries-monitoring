import numpy as np
from sklearn import linear_model
import os, json

path_to_json = 'data/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print (json_files)

for js in json_files:
    with open(os.path.join(path_to_json, js)) as json_file:
        models = []
        models.append(json.load(json_file))

results = []
for img in models[0].keys():
    resultforthiskey = []
    for model in models:
        resultforthiskey.extend(model[img])
        results.append(resultforthiskey)
        
y = [0,0,1,0,0,0,0]
m = linear_model.LogisticRegression()
m1 = m.fit(results,y)