# TicketWise - Developer Walkthrough

This is a writeup of how I built the project, what I was thinking at each step, and what went wrong along the way.

---

## The Problem

So basically, companies get a lot of support tickets. Customers email in saying "I can't log in" or "I got charged twice" or "the app is slow." Someone on the support team has to read each one, figure out what type of problem it is, and write a reply. That takes time.

This model does both of those things. It reads a new ticket and guesses the category using a simple ML model. Then if the agent wants, they click a button and an LLM writes a draft reply for them. The agent can edit it before sending. And if the LLM is down or the API key doesn't work, the ML part keeps working on its own. No crashes.

This is an internal tool for the support team. The customer never sees this screen.

---

## Before Writing Code

I went through the spec twice. First time just to get the overall picture. Second time I made a mental list of everything I'd need to build.

I noticed the project is really two things happening at different times. There's a one-time setup: make the data, clean it, train models, pick the best one, save it. And then there's the actual app: load the saved model, let someone type a ticket, predict, show results, call the LLM.

Made sense to build them in that order because the app needs the model to exist first.

I also didn't want to put everything in one giant file. The training code and the UI code do completely different things. So I split them up early.

---

## The Files and Why They Exist

```
TicketWise/
├── generate_data.py    - makes the fake ticket dataset
├── train_model.py      - trains and compares both models
├── llm_service.py      - talks to the Gemini API
├── app.py              - the streamlit app (the actual UI)
├── config.py           - settings like threshold, model name, file paths
├── utils.py            - the text cleaning function
├── data/tickets.csv    - the dataset (made by generate_data.py)
├── models/             - saved model files (made by train_model.py)
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── screenshots/
```

config.py exists because I originally had settings hardcoded in random places. The confidence threshold was buried in app.py, the Gemini model name was in llm_service.py. When I had to change the model name (more on that later), I had to dig through files to find it. So I just put all the settings in one file. Way easier.

utils.py has just one function, clean_text. Sounds overkill but both train_model.py and app.py need it. The ticket at runtime has to be cleaned the exact same way as the training data or the TF-IDF columns won't match and predictions will be garbage. I tried importing it from train_model.py into app.py first but that felt weird. The app shouldn't depend on the training script. So I put the function in its own file.

llm_service.py is separate because all the API stuff (prompt, call, parse, error handling) is one job. If I need to switch from Gemini to OpenAI later, I just change this one file.

---

## Building It

### Making the Dataset

The spec lets me create synthetic data. 200-300 tickets, 5 categories. So I wrote templates for each category. Like for Access Issue I have things like "Cannot log into my account" with a description about invalid credentials. For Billing I have "Charged twice this month" and so on.

The function picks random templates, sticks on a random extra sentence like "please help as soon as possible" or "this has been going on for two days," picks a random urgency level, and does that 50 times per category. 5 x 50 = 250 tickets.

But here's the thing. When I first ran it, both models got 100% accuracy. Every single prediction was right. That's because my templates were too clean. The word "login" only ever appeared in Access Issue tickets. "Refund" only in Billing. TF-IDF could separate them perfectly every time.

That's a problem because 100% on synthetic data doesn't prove anything. So I did two things. First I added some confusing templates on purpose, like "Login page is extremely slow" which could be Access or Performance. Second I added label noise. I randomly flip the category on about 6% of tickets. This simulates what happens in real life when a human agent accidentally puts a ticket in the wrong category. Everyone does it.

After that, accuracy came down to 94%. The confusion matrix actually shows some misclassifications now. Much more realistic.

### Cleaning the Text

Pretty basic. I lowercase everything, remove punctuation and special characters with regex, and collapse multiple spaces into one.

I didn't do stemming because it turns "credentials" into "credenti" and the app shows the user which words influenced the prediction. That needs to be readable. I also skipped lemmatisation because it needs extra libraries (NLTK or spaCy) and doesn't really help much on 250 short tickets.

### TF-IDF

This is the part that turns words into numbers so the model can work with them.

I set max_features to 3000 to keep the vocabulary from getting too big. ngram_range is (1, 2) which means it captures individual words and two-word phrases. So "password" and "password reset" are both features. The phrase often carries more meaning than either word alone.

I also set stop_words="english" to remove common words like "the", "is", "to."

I actually forgot stop_words at first. The app was showing "top influencing words: to, takes, it, the, to be." Completely useless. Spotted it immediately when I looked at the UI. Added stop_words, retrained, and the words became things like "dashboard, takes, instant." Much better.

One thing that's really important here: the vectorizer learns the vocabulary only from the training data (fit_transform). When I process new tickets or the test set, I only use transform without fit. If I re-fitted on the test set, it would learn a different vocabulary and the columns wouldn't match. That's data leakage.

### Training the Models

The spec asks for Naive Bayes and Logistic Regression. They actually make a good pair because they work differently.

Naive Bayes looks at how words are distributed within each category. Like "login" appears a lot in Access Issue tickets, so if a new ticket has "login" in it, NB leans toward Access Issue.

Logistic Regression is different. It learns a weight for each word toward each category. "Login" might get a weight of +2.3 for Access Issue and -1.5 for Billing. It adds up all the weights and picks the highest scoring category.

I split the data 80/20 for train/test. I used stratify=y in the split so each category has roughly equal representation in both sets. Without that, you could accidentally end up with all the Access Issue tickets in training and none in test.

Both models got 94% accuracy and 93.86% F1. They tied. That makes sense honestly. 50 test tickets, fairly clean features, both linear classifiers. They ended up making the exact same predictions. The code picks Naive Bayes when there's a tie just because Python's max() returns the first thing it sees.

### Saving Everything

I save three files. The model itself, the vectorizer, and the metrics. The model and vectorizer are a pair. You can't use one without the other. The vectorizer knows which word maps to which column and what the IDF values are. Without it you can't transform a new ticket properly.

I used joblib for saving because it handles numpy arrays better than pickle and it's what scikit-learn recommends.

The metrics go into a JSON file so the app can display them without recomputing anything.

### The Streamlit App

The app has two tabs. One for classifying tickets (the main thing), one for viewing the model evaluation metrics.

I load the model and vectorizer once using @st.cache_resource. Without that, every single click would reload them from disk. They're small files so it'd still be fast, but it's wasteful.

For the confidence threshold I went with 60%. My thinking: with 5 categories, pure random guessing gives you 20%. So 60% means the model is 3x more confident than just guessing randomly. That felt like a reasonable line. Below that, I show a warning saying "confidence is low, maybe have a human check this." In a real setting you'd tune this number based on how many tickets you're willing to send for manual review.

The top influencing words part works by looking at which words from the ticket have the highest weights in the model for the predicted category. For NB that's log-probabilities. For LR that's the learned coefficients. It picks the top 5 and displays them so the agent can see why the model made that prediction.

**A bug I spent time on:** when I clicked "Generate assisted response," the prediction above it just disappeared. Gone. Took me a while to understand why. Streamlit reruns the entire script every time you click anything. The prediction was being computed inside the form's submit block. When a different button was clicked, the form wasn't resubmitted, so that block was skipped, and the prediction variables just didn't exist anymore. I fixed it by saving everything to st.session_state. That persists between reruns.

### LLM Integration

When the agent clicks the generate button, I build a prompt with the ticket text, the predicted category, confidence, and urgency. I tell the LLM to return only JSON with three keys: issue_summary, customer_response, internal_action. I ask for JSON because I need to display each piece separately in the UI. If it returned a paragraph I'd have to parse it with regex which breaks easily.

The entire API call is wrapped in try/except. If it fails for any reason, bad key, no internet, rate limit, whatever, it returns {"ok": False} and the app shows an error message. But the ML prediction above stays right where it is. That's the whole point of requirement 11. The ML and LLM parts are independent. One breaking doesn't affect the other.

This actually happened to me during development. Gemini kept deprecating models on me. I started with gemini-2.0-flash and got a weird quota error where the limit was literally 0. Switched to gemini-2.5-flash and got a 404 saying it's "no longer available to new users." I ended up using gemini-flash-latest which is just an alias that points to whatever the current model is. That finally worked.

For the refinement feature, if the agent types something like "make it shorter" or "add a request for browser details," I send the previous LLM response plus that instruction as a new prompt. Fresh API call each time.

---

## Test Scenarios

**High confidence:**
Subject: "Dashboard takes forever to load"
Description: "The main dashboard takes over thirty seconds to load, it used to be instant. This is affecting my whole team."
Urgency: High
Should predict Performance Issue with high confidence. No warning. Meaningful top words.

**Low confidence:**
Subject: "Need help"
Description: "Something is not right please check"
Urgency: Medium
Should have low confidence and show the manual review warning.

**LLM failure:**
Break the API key in .env, submit any ticket, click Generate.
Should show a red error for the LLM section but the ML prediction above should still be there and working fine.
