# TicketWise

ML support ticket classifier with an LLM-assisted response layer.
TicketWise is a Machine Learning based support ticket classifier with an LLM-assisted response generator. 
It predicts the category of a support ticket and can generate a suggested response using Gemini.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY (get one free at aistudio.google.com/apikey)
```

## Run

```bash
python generate_data.py   # creates data/tickets.csv (skip if you already have a dataset)
python train_model.py     # trains, compares, and saves the best model to models/
streamlit run app.py      # launches the UI
```

## Project structure

```
TicketWise/
├── data/tickets.csv         # synthetic dataset
├── models/                  # saved model + vectorizer + metrics (created by train_model.py)
├── generate_data.py         # builds the synthetic dataset
├── train_model.py           # cleans data, TF-IDF, trains + compares NB and LogReg
├── llm_service.py           # talks to the LLM (generate + refine)
├── app.py                   # Streamlit UI
├── config.py                # thresholds, model name, file paths
├── utils.py                 # shared text-cleaning function
├── requirements.txt / README.md / .env.example / .gitignore
└── screenshots/
```

## How it works

1. `generate_data.py` builds a synthetic labelled dataset of support tickets.

2. `train_model.py` cleans the text, builds TF-IDF features, trains Naive Bayes and
   Logistic Regression, compares them on accuracy/precision/recall/F1 and a confusion
   matrix, and saves the better-performing model.

3. `app.py` is the Streamlit app: shows the evaluation metrics, takes a new ticket,
   predicts its category with a confidence score and the top influencing words, warns on
   low confidence, and (if an API key is configured) asks an LLM for a customer-facing
   response, an issue summary, and a recommended internal action. The response can be
   refined with a follow-up instruction. If the LLM call fails, the ML prediction still
   displays normally.

## Test scenarios

**Scenario 1 — high-confidence performance issue**
- Subject: "Dashboard takes forever to load"
- Description: "The main dashboard takes over thirty seconds to load, it used to be
  instant. This is affecting my whole team."
- Urgency: High
- Expected: predicted category "Performance Issue" with high confidence, no manual-review
  warning, LLM returns a summary/response/action referencing slow load times.

**Scenario 2 — ambiguous/low-confidence ticket**
- Subject: "Something is wrong"
- Description: "It's not working properly and I don't know why."
- Urgency: Medium
- Expected: low confidence score, manual-review warning displayed, LLM response still
  generated (or skipped if the API key is missing/invalid).
