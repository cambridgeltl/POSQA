from sklearn.metrics import f1_score

original_preds = []
contextual_preds = []

general_gold_labels = []
special_gold_labels = []

with open("output/generalQ_gold_labels.txt.txt", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        general_gold_labels.append(line.strip())

with open("output/specialQ_gold_labels.txt.txt", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        special_gold_labels.append(line.strip())

with open("output/GPT3_general_pureQ_pred_labels.txt", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        original_preds.append(line.strip())

with open("output/GPT3_general_TS_pred_labels.txt", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        contextual_preds.append(line.strip())

# with open("output/GPT3_special_pureQ_pred_labels.txt", 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         original_preds.append(line.strip())

# with open("output/GPT3_special_TS_pred_labels.txt", 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         contextual_preds.append(line.strip())

def cal_accuracy(pred_labels, labels):
    acc = sum([int(i==j) for i,j in zip(pred_labels, labels)])/len(pred_labels)
    return acc

def cal_reverse_accuracy(pred_labels, labels):
    acc = sum([int(i!=j) for i,j in zip(pred_labels, labels)])/len(pred_labels)
    return acc

def cal_results(original_pred_results, contextual_pred_results, gold_labels):
    original_true_predicted = 0
    original_false_predicted = 0
    original_true_contextual_false_predicted = 0
    original_false_contextual_true_predicted = 0

    for i in range(len(original_preds)):
        if original_preds[i] == gold_labels[i]:
            original_true_predicted += 1
            if contextual_preds[i] != gold_labels[i]:
                original_true_contextual_false_predicted += 1
        else:
            original_false_predicted += 1
            if contextual_preds[i] == gold_labels[i]:
                original_false_contextual_true_predicted += 1
    
    print('Original Accuracy: ', cal_accuracy(original_preds, gold_labels))
    print('Original Micro F1:', f1_score(original_preds, gold_labels, average='micro'))
    print('Original Macro F1:', f1_score(original_preds, gold_labels, average='macro'))
    print('Contextual Accuracy: ', cal_accuracy(contextual_preds, gold_labels))
    print('Contextual Micro F1:', f1_score(contextual_preds, gold_labels, average='micro'))
    print('Contextual Macro F1:', f1_score(contextual_preds, gold_labels, average='macro'))
    print('#original_true_predicted: ', original_true_predicted)
    print('#original_true_contextual_false_predicted: ', original_true_contextual_false_predicted)
    print('#original_false_predicted: ', original_false_predicted)
    print('#original_false_contextual_true_predicted: ', original_false_contextual_true_predicted)
    print('Contextual Effective Rate: ', original_false_contextual_true_predicted/original_false_predicted)
    print('Contextual Misleading Rate: ', original_true_contextual_false_predicted/original_true_predicted)

def cal_reverse_results(original_pred_results, contextual_pred_results, gold_labels):
    original_true_predicted = 0
    original_false_predicted = 0
    original_true_contextual_false_predicted = 0
    original_false_contextual_true_predicted = 0

    for i in range(len(original_preds)):
        if original_preds[i] == gold_labels[i]:
            original_true_predicted += 1
            if contextual_preds[i] != gold_labels[i]:
                original_true_contextual_false_predicted += 1
        else:
            original_false_predicted += 1
            if contextual_preds[i] == gold_labels[i]:
                original_false_contextual_true_predicted += 1

    print('Original Accuracy: ', cal_accuracy(original_preds, gold_labels))
    print('Original Macro F1:', f1_score(original_preds, gold_labels, average='macro'))
    print('Contextual Accuracy: ', cal_reverse_accuracy(contextual_preds, gold_labels))
    print('Contextual Macro F1:', f1_score(contextual_preds, gold_labels, average='macro'))
    print('#original_true_predicted: ', original_true_predicted)
    print('#original_true_contextual_false_predicted: ', original_true_contextual_false_predicted)
    print('#original_false_predicted: ', original_false_predicted)
    print('#original_false_contextual_true_predicted: ', original_false_contextual_true_predicted)
    print('Contextual Effective Rate: ', original_false_contextual_true_predicted/original_false_predicted)
    print('Contextual Misleading Rate: ', original_true_contextual_false_predicted/original_true_predicted)

print('-------------------------------------------------------------')
cal_results(original_preds, contextual_preds, general_gold_labels)
# cal_reverse_results(original_preds, contextual_preds, general_gold_labels)
# cal_results(original_preds, contextual_preds, special_gold_labels)
# cal_reverse_results(original_preds, contextual_preds, special_gold_labels)
print('-------------------------------------------------------------')
