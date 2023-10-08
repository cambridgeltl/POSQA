import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

with open("dataset/size_official.json", 'r', encoding='utf-8') as f:
    size = json.load(f)

with open("dataset/ents.json", 'r', encoding='utf-8') as f:
    ents = json.load(f)

with open("knowledge/ents_general_knowledge.json", 'r', encoding='utf-8') as f:
    ents_knowledge = json.load(f)

with open("knowledge/ents_size_knowledge.json", 'r', encoding='utf-8') as f:
    ents_size_knowledge = json.load(f)

ents_scale = {}
for k,v in ents.items():
    ents_scale[v['name']] = ents[k]['scale']

ents_size = {}
for k,v in ents.items():
    ents_size[v['name']] = ents[k]['size']

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl", device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")

preds = []
labels = []

for k, v in tqdm(size.items()):
    labels.append(v['special_question_label'])
    try:
        input_text = 'The size of '+ v['h_name'] + ' is ' + ents_scale[v['h_name']] + '. ' + 'The size of '+ v['t_name'] + ' is ' + ents_scale[v['t_name']] + '. ' + v['special_question'] + "? Answer this question with entity name."
    except Exception as e:
        print(repr(e))
        input_text = v['special_question'] + '?'
    inputs = tokenizer.encode(input_text, max_length=768, truncation=True, return_tensors="pt").cuda()
    outputs = model.generate(inputs, max_length=768)
    pred = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    preds.append(pred)

with open("output/FLAN_T5_XLL_special_TS_pred.txt","w") as output_file:
    for line in preds:
        output_file.write(line + '\n')

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

with open("output/FLAN_T5_XLL_special_TS_pred_labels.txt","w") as output_file:
    for line in pred_labels:
        output_file.write(str(line) + '\n')

acc = sum([int(i==j) for i,j in zip(pred_labels, labels)])/len(pred_labels)
print('Accuracy: ', acc)
