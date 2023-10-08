import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
    labels.append(v['general_question_label'])
    try:
        input_text = 'The size of '+ v['h_name'] + ' is ' + ents_scale[v['h_name']] + '. ' + 'The size of '+ v['t_name'] + ' is ' + ents_scale[v['t_name']] + '. ' + 'Question: ' + v['general_question'] + "? Answer this question with yes or no."
    except Exception as e:
        print(repr(e))
        input_text = v['general_question'] + '?'
    inputs = tokenizer.encode(input_text, max_length=768, truncation=True, return_tensors="pt")
    outputs = model.generate(inputs, max_length=768)
    pred = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    preds.append(pred)

with open("output/FLAN_T5_XXL_general_TS_pred.txt","w") as output_file:
    for line in preds:
        output_file.write(line + '\n')
    
pred_labels = []
error_index = []
for i, pred in enumerate(preds):
    if 'yes' in pred:
        pred_labels.append(True)
    elif 'Yes' in pred:
        pred_labels.append(True)
    elif 'no' in pred:
        pred_labels.append(False)
    elif 'No' in pred:
        pred_labels.append(False)
    else:
        error_index.append(i)
        print('Format Error at Index ', i)

with open("output/FLAN_T5_XXL_general_TS_pred_labels.txt","w") as output_file:
    for line in pred_labels:
        output_file.write(str(line) + '\n')
        
with open("output/error_index.txt","w") as output_file:
    for line in error_index:
        output_file.write(str(line) + '\n')
        
if not error_index:       
    acc = sum([int(i==j) for i,j in zip(pred_labels, labels)])/len(pred_labels)
    print('Accuracy: ', acc)
else:
    print('check the error index file')
