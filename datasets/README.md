# Generated Knowledge

This file contains the generated the entity-related knowledge statement by querying GPT-3. 

`ents.json` is the information of each entity, which contains the entity name, the entity type, the entity description, and the entity size.

`posqa.json` is the POSQA datasets, which contains the entity pair, the entity pair type, the entity pair size, the general question, the general question label, the special question, and the special question label.

`posqa_smaller.json` and `posqa_bigger.json` are the POSQA datasets contains only smaller and bigger questions, respectively, which is designed for the ablation study.

# Example
```
"82_2_0": {
    "h_name": "Yellow fever virus",
    "t_name": "Local Group of Galaxy",
    "magnitude": [
      "-8",
      "22",
      30
    ],
    "general_question": "Is the Yellow fever virus bigger than the Local Group of Galaxy",
    "general_question_label": false,
    "special_question": "Which one is bigger between Yellow fever virus and Local Group of Galaxy",
    "special_question_label": "Local Group of Galaxy"
  }
```

- `h_name` is the name of the head entity.
- `t_name` is the name of the tail entity.
- `magnitude` is the magnitude of the head entity and the tail entity.
- `general_question` is the general question.
- `general_question_label` is the label of the general question.
- `special_question` is the special question.
- `special_question_label` is the label of the special question.
