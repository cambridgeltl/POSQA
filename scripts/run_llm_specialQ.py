import os
import openai

openai.api_key = "API-key"

import time
import json
import functools
from tqdm import tqdm

with open("dataset/size_official.json", 'r', encoding='utf-8') as f:
    size = json.load(f)

with open("dataset/ents.json", 'r', encoding='utf-8') as f:
    ents = json.load(f)

with open("knowledge/ents_size_knowledge.json", 'r', encoding='utf-8') as f:
    ents_size_knowledge = json.load(f)

with open("knowledge/ents_general_knowledge.json", 'r', encoding='utf-8') as f:
    ents_general_knowledge = json.load(f)

ents_scale = {}
for k,v in ents.items():
    ents_scale[v['name']] = ents[k]['scale']

@functools.lru_cache()
def get_completion(prompt):
    got_result = False
    while not got_result:
        try:
            # for GPT-3
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=512,
                temperature=0
                )
            # for ChatGPT
            # response = openai.ChatCompletion.create(
            #     max_tokens=512,
            #     model="gpt-3.5-turbo",
            #     messages = [{"role": "user", "content": prompt}],
            #     temperature=0
            #     )
            got_result = True
        except Exception:
            time.sleep(1)
    return response

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

preds = []
labels = []

for k, v in tqdm(size.items()):
	labels.append(v['special_question_label'])
	prompt = 'The size of '+ v['h_name'] + ' is ' + ents_scale[v['h_name']] + '. ' + 'The size of '+ v['t_name'] + ' is ' + ents_scale[v['t_name']] + '. ' + v['special_question'] + "? Answer this question with entity name."
	response = get_completion(prompt)
	# for GPT-3
	preds.append(response["choices"][0]["text"])
	# for ChatGPT
	# preds.append(response["choices"][0]["message"]["content"])

all_h_name = []
all_t_name = []
for k,v in size.items():
	all_h_name.append(v['h_name'])
	all_t_name.append(v['t_name'])

pred_labels = []
for i, pred in enumerate(preds):
	if 'bigger' in pred:
		pred_labels.append(all_h_name[i])
	elif 'smaller' in pred:
		pred_labels.append(all_t_name[i])
	elif levenshteinDistance(pred, all_h_name[i]) <= levenshteinDistance(pred, all_t_name[i]):
		pred_labels.append(all_h_name[i])
	else:
		pred_labels.append(all_t_name[i])

with open("output/GPT3_special_TS_preds.txt","w") as output_file:
    for line in preds:
        output_file.write(line + '\n')

with open("output/GPT3_special_TS_pred_labels.txt","w") as output_file:
    for line in pred_labels:
        output_file.write(str(line) + '\n')
        
acc = sum([int(i==j) for i,j in zip(pred_labels, labels)])/len(pred_labels)
print('Accuracy: ', acc)
