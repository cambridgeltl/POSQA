# Setup
```
pip install -r requirements.txt
```

# Scripts to Run the Experiments

`run_flan_t5_generalQ.py` and `run_flan_t5_specialQ.py` are the scripts to run experiments with [FLAN-T5](https://huggingface.co/google/flan-t5-xxl/) models on general questions and special questions, respectively.

`run_llm_generalQ.py` and `run_llm_specialQ.py` are the scripts to run experiments with LLMs (i.e., ChatGPT) on general questions and special questions, respectively. Please replace `"API-key"` in the files with your own OpenAI API key.

`cal_metric_scores.py` is used to calculate the automatic evaluation results.