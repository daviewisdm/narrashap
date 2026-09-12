# narrashap

**Turn confusing model predictions into plain English — in one line of code.**

If you've ever built a machine learning model and had someone ask "okay but *why* did it say that?", this library is for that moment. `narrashap` takes the technical output of a popular explainability tool called **SHAP** and turns it into a sentence a regular person can actually read — no statistics background required.

This guide assumes you're new to this. It walks through everything: getting the code, installing it, running your first example, and setting up a free AI service to make the explanations even better.

---

## What problem does this solve?

Machine learning models (like one that predicts a health risk, or flags a fraudulent transaction) don't explain themselves. A tool called **SHAP** can tell you *which* factors influenced a prediction and by how much — but its raw output looks like this:

```
Age: +0.31
Race_Black: +1.58
Family_History: -0.57
```

That's useful to a data scientist. It means nothing to a patient, a manager, or anyone without a machine learning background.

`narrashap` turns that into something like:

> "Based on the information provided, the estimated risk is 65.6%. This falls in the moderate risk range. The biggest factor was race, which raised the estimated risk. Another important factor was family history, which lowered the estimated risk. This is only a computer estimate based on patterns in data. It does not mean any one factor by itself causes the outcome, and it is not a diagnosis."

That's the whole point of this project.

---

## Before you start: things you'll need

- **Python 3.10 or newer** installed on your computer. Check with:
  ```bash
  python --version
  ```
- **Git**, to download (clone) the code. Check with:
  ```bash
  git --version
  ```
- A trained machine learning model you want to explain, along with the data it was trained on. (If you don't have one yet, this library has nothing to explain — it works *on top of* a model, it doesn't build one for you.)
- (Optional, but recommended) A free **Groq** API key, if you want richer, more natural-sounding explanations instead of the simpler built-in template sentences. More on this below — it's free and takes two minutes.

---

## Step 1: Clone the repository

"Cloning" just means downloading a copy of the code from GitHub onto your computer.

Open a terminal (Command Prompt, PowerShell, or Terminal on Mac) and run:

```bash
git clone git@github.com:daviewisdm/narrashap.git
```

If that gives you an error about SSH keys or permissions, use this version instead (it works without any extra setup):

```bash
git clone https://github.com/daviewisdm/narrashap.git
```

Then move into the folder it just created:

```bash
cd narrashap
```

---

## Step 2: Install it

From inside that `narrashap` folder, run:

```bash
pip install -e .
```

**What this does:** it installs `narrashap` so Python can find it, in "editable" mode — meaning if you (or someone else) later change the code, you don't have to reinstall it every time.

**If you're on Windows and see an error mentioning "externally managed environment":** add this flag instead:

```bash
pip install -e . --break-system-packages
```

**To check it worked**, run this:

```bash
python -c "from narrashap import narrate; print('It works!')"
```

If you see `It works!` printed, you're good to move on. If you see an error instead, check the [Troubleshooting](#troubleshooting) section near the end.

---

## Step 3: Try it with a tiny example (no model needed yet)

Before wiring this into your real project, let's prove it works with fake data. Create a file called `quick_test.py` (or a new cell in a Jupyter notebook) and paste this in:

```python
import pandas as pd
from narrashap import narrate

# Fake "training data" — pretend this is what your model was trained on
training_data = pd.DataFrame({
    "Age": [25, 30, 45, 60, 35, 50],
    "Smoker": [0, 0, 1, 1, 0, 1],
})

# Fake SHAP values for one person: Age contributed +0.4, Smoker contributed +0.9
shap_values = [0.4, 0.9]

# The actual feature values for this one person
instance = pd.Series({"Age": 45, "Smoker": 1})

result = narrate(
    shap_values=shap_values,
    instance=instance,
    training_data=training_data,
    feature_names=["Age", "Smoker"],
    base_value=0.1,          # the model's "starting point" before considering this person
    risk_percentage=72.0,    # optional: the real percentage you'd show a user
    risk_level="HIGH RISK",  # optional: a label to go with it
)

print(result)
```

Run it:

```bash
python quick_test.py
```

You should see a full paragraph explaining that the estimated risk was 72%, that being a smoker was a strong factor, and so on — all without needing any AI service or API key. This is `narrashap`'s **free, offline mode** — it uses fixed sentence templates, not an AI model, so it always works with zero setup and zero cost.

---

## Step 4 (optional but recommended): Set up Groq for richer explanations

The example above works, but the sentences follow a fairly repetitive pattern since they're built from fixed templates. If you want explanations that read more naturally — more like a person actually wrote them — you can plug in a free AI service called **Groq**.

### Why Groq specifically?

It's free (no credit card needed to start), fast, and gives you a generous number of free requests per day — more than enough for testing and even regular use.

### 4a. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com) and sign up (free).
2. Once logged in, find the **API Keys** section and create a new key.
3. Copy the key — it will look something like `gsk_abc123...`. You won't be able to see it again after you close that page, so copy it somewhere safe now.

### 4b. Install the Groq Python package

```bash
pip install groq
```

### 4c. Make your API key available to your code

Your code needs to know your key, but you should **never** type your key directly into a file that gets shared or uploaded to GitHub — that's a real security risk. Instead, set it as an "environment variable" for your terminal session.

**On Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY = "gsk_your-actual-key-here"
```

**On Mac/Linux:**
```bash
export GROQ_API_KEY="gsk_your-actual-key-here"
```

⚠️ **Important:** this only lasts for your *current terminal window*. If you close the terminal and open a new one, you'll need to set it again — that's normal, not a bug. If you find that annoying, see the "make it permanent" note below.

**To check it's actually set**, run:

```powershell
echo $env:GROQ_API_KEY      # Windows PowerShell
```
```bash
echo $GROQ_API_KEY          # Mac/Linux
```

If that prints your key back, you're set. If it prints nothing, the step above didn't take — try it again in the exact same terminal window you'll run your code from.

**Making it permanent (Windows only), so you don't have to repeat this every time:**
```powershell
setx GROQ_API_KEY "gsk_your-actual-key-here"
```
Note: after running `setx`, you must close your terminal completely and open a brand new one before it takes effect.

### 4d. Use it in your code

Now, instead of the free template mode, pass `llm_client=GroqClient()`:

```python
from narrashap.core.llm_client import GroqClient

result = narrate(
    shap_values=shap_values,
    instance=instance,
    training_data=training_data,
    feature_names=["Age", "Smoker"],
    base_value=0.1,
    risk_percentage=72.0,
    risk_level="HIGH RISK",
    llm_client=GroqClient(),   # <-- this line switches on the AI-generated version
)

print(result)
```

Run it again — this time it'll make a real (free) call to Groq's servers and give you back a longer, more natural-sounding paragraph.

**If you get an error saying a "model" doesn't exist or "model_not_found":** AI companies frequently retire old models and replace them with newer ones. Check Groq's current model list at [console.groq.com/docs/models](https://console.groq.com/docs/models) and update the model name if needed — see the [Troubleshooting](#troubleshooting) section for exactly where to change this.

---

## Step 5: Using it with a real model

Once you have an actual trained model, here's the general shape of what you need:

```python
import shap
import pandas as pd
from narrashap import narrate

# 1. Your already-trained model and the data it learned from
#    (however you normally load these — pickle, joblib, etc.)
model = ...          # your trained model
X_train = ...         # the data you trained it on (a pandas DataFrame)
X_instance = ...      # the ONE row/person you want to explain

# 2. Compute SHAP values for that one instance
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_instance)[0]

# 3. Get the narrative — that's it
result = narrate(
    shap_values=shap_values,
    instance=X_instance.iloc[0],
    training_data=X_train,
    risk_percentage=65.6,     # your model's actual predicted probability, as a %
    risk_level="MODERATE RISK",
)

print(result)
```

**A real, working example of this exact pattern** — including handling a scikit-learn `Pipeline`, a preprocessing step, and a logistic regression model specifically — lives in the [DadaCare project](#) (`uf-risk-model` repo), in `fibroids_hospital_app/app.py`. If your model setup looks similar (a `Pipeline` with a `preprocessor` step and a classifier step), that file is the best reference for exact code to copy.

---

## Using this in a Jupyter Notebook (ready-to-paste cell)

If you're working in a notebook rather than a script, here's a self-contained cell you can paste in and adapt. It assumes you already have, from earlier cells in your notebook: a fitted `model` (a scikit-learn `Pipeline` with a `preprocessor` step and a `classifier` step), and `X_train` (the DataFrame the model was trained on). **It does not train or fit anything new** — it only reads from what your notebook has already built.

```python
# --- narrashap: explain one prediction, right here in the notebook ---
from narrashap import narrate

# Pick which row to explain — change this to any row in your data
patient_idx = 0
X_instance = X_train.iloc[[patient_idx]]

# Reuse your notebook's already-fitted preprocessor and model
preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']

X_trans = preprocessor.transform(X_instance)
feature_names = list(preprocessor.get_feature_names_out())

# Clean up names like "num__Age" -> "Age" so the narrative reads naturally
def clean_name(name):
    return name.replace("num__", "").replace("cat__", "").replace("_", " ").title()

clean_feature_names = [clean_name(fn) for fn in feature_names]

# SHAP values for a logistic regression: coefficient * feature value
# (swap this line out if your model isn't logistic regression)
shap_vals = classifier.coef_[0] * X_trans[0]

# Training data, transformed the same way, so narrate() can compute
# percentile context ("higher than X out of 100 in the data")
X_train_trans = preprocessor.transform(X_train)
training_df = pd.DataFrame(X_train_trans, columns=clean_feature_names)

# The model's real predicted probability, as a percentage
proba = classifier.predict_proba(X_trans)[0][1]
risk_pct = round(proba * 100, 1)
risk_level = "HIGH RISK" if risk_pct >= 70 else "MODERATE RISK" if risk_pct >= 40 else "LOW RISK"

# The one-liner that ties it all together
narrated_shap = narrate(
    shap_values=shap_vals,
    instance=pd.Series(X_trans[0], index=clean_feature_names),
    training_data=training_df,
    feature_names=clean_feature_names,
    base_value=classifier.intercept_[0],
    risk_percentage=risk_pct,
    risk_level=risk_level,
)

print(narrated_shap)
```

**Want the AI-generated version instead of the free template?** Add one line:

```python
from narrashap.core.llm_client import GroqClient

narrated_shap = narrate(
    shap_values=shap_vals,
    instance=pd.Series(X_trans[0], index=clean_feature_names),
    training_data=training_df,
    feature_names=clean_feature_names,
    base_value=classifier.intercept_[0],
    risk_percentage=risk_pct,
    risk_level=risk_level,
    llm_client=GroqClient(),  # needs GROQ_API_KEY set — see Step 4 above
)

print(narrated_shap)
```

If your notebook's variable names don't match (`model`, `X_train`, or the pipeline's step names), just swap them in — everything else stays the same.

---

## What you get back

By default, `narrate()` gives you back a plain string — just the explanation text, ready to print or display.

If you want more detail — like a "how confident is this explanation" score — add `return_details=True`:

```python
result = narrate(
    shap_values=shap_values,
    instance=X_instance.iloc[0],
    training_data=X_train,
    return_details=True,
)

print(result.text)              # the explanation, same as before
print(result.fidelity_score)    # a 0-1 score of how well the text matches the real numbers
```

Note: `fidelity_score` is only calculated when you're using Groq (or another AI backend) — it doesn't apply to the free template mode, so it'll show up as `None` there.

---

## Understanding the safety guardrails (why this matters)

This library is deliberately careful about **not implying that a factor *causes* an outcome** — it only ever says a factor is "associated with" or "one of the strongest factors in" a prediction. This distinction matters a lot in fields like healthcare: SHAP can tell you a factor influenced a model's prediction, but it can never prove that factor *causes* anything in real life.

Because of this, every generated explanation is automatically checked for risky phrasing (words like "causes," "proven," "guaranteed") before it's shown to you. If the AI-generated version accidentally uses one of these words, the library automatically asks it to rewrite the sentence — and if it still can't produce a safe version after one retry, it will raise an error rather than quietly showing you something misleading. This is intentional — treat that error as the system doing its job, not a bug to route around.

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'narrashap'"**
You probably installed it in a different Python environment than the one you're running your code in. If you're using a virtual environment or conda environment, make sure it's activated *before* you run `pip install -e .`, and activated again every time you come back to work on this.

**"narrashap is not installed in this environment" (inside a Jupyter notebook)**
Notebooks sometimes use a different Python environment than your terminal. Run this in a notebook cell to check:
```python
import sys
print(sys.executable)
```
Then run `pip install -e /path/to/narrashap` using that *exact* Python (you may need to run it as a notebook cell with `!` in front: `!pip install -e /path/to/narrashap`).

**"Groq API key required" even though I set it**
Environment variables reset every time you open a new terminal window (unless you used `setx` on Windows and reopened the terminal afterward). Set it again in the terminal window you're actually running your code from.

**"The model `...` does not exist or you do not have access to it" (from Groq)**
The specific AI model name is outdated. Open `narrashap/core/llm_client.py`, find the `GroqClient` class, and update the default `model=` value to a current model name from [console.groq.com/docs/models](https://console.groq.com/docs/models).

**The AI-generated explanation sounds too technical / mentions raw numbers**
Make sure you're passing `risk_percentage` and `risk_level` into `narrate()`. Without them, the explanation falls back to showing the model's raw internal numbers, which aren't meant for a general audience.

**Everything crashes with a "Generated narrative still contains banned phrases after retry" error**
This means the AI-generated text used risky causal language twice in a row and the safety check correctly blocked it. This is rare, and usually resolves itself if you just try again (AI responses vary each time). If it happens constantly, this is worth reporting as an issue.

---

## Running the tests

If you want to confirm everything on your machine is working correctly:

```bash
python -m pytest -q
```

You should see all tests pass. If any fail, don't ignore it — that usually means something in your local setup differs from what's expected (wrong Python version, missing dependency, etc.).

---

## Project structure, for anyone curious

```
narrashap/
├── narrashap/
│   ├── convenience.py   # the narrate() one-liner — start here
│   ├── core/
│   │   ├── extractor.py   # turns raw SHAP output into a clean structure
│   │   ├── llm_client.py  # the different AI backends (Groq, Anthropic, or none)
│   │   ├── narrator.py    # builds the explanation and checks it for safety
│   │   └── scorer.py      # scores how accurate a generated explanation is
│   ├── domains/           # different "flavors" for different industries (healthcare, fraud, ...)
│   └── templates/         # the free, no-AI-needed sentence templates
└── tests/                 # automated tests confirming everything works
```

---

## Current limitations (being upfront about what this can't do yet)

- This has been thoroughly tested against a **logistic regression** healthcare model. Other model types (tree-based models, neural networks) should work in principle but haven't been confirmed.
- Only "healthcare" and "fraud" have real terminology and tone configured so far — other industries (finance, insurance) are planned but not built yet.
- It currently handles one prediction at a time, not a whole batch of predictions at once.

---

## License

TBD.