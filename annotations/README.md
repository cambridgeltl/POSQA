# Human Annotations

This folder contains the human annotations on the sampled POSQA dataset. 

`annotation_offline\` contains the annotations on the offline sampled POSQA dataset.

`annotation_online\` contains the annotations on the online sampled POSQA dataset.

# Example

```
{"id": "60", 
"displayed_text": "Is the Local Group of Galaxy bigger than the Sesame seed?", 
"label_annotations":
     {"affordance": {"Yes": "true"}}, 
"span_annotations":
     {},
"behavioral_data": 
     {"time_string": "Time spent: 0d 0h 0m 5s "}}
```

- `id` is the id of the question.
- `displayed_text` is the question.
- `label_annotations` is the human annotation of the question.
- `behavioral_data` is the time spent on the question.