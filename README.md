# Recoverly

**Recovery decisions with evidence — an AI decisioning system for recovering revenue from failed payments.**

`Track: AI Revenue Recovery` · `Solo build` · `Status: MVP`

Built solo by Pranav for the Razorpay Buildathon.

> - Demo video:https://drive.google.com/file/d/1N5HzJ2iBJLro6P7xMF8PzdxjbyGDh_PS/view?usp=sharing


## Contents

- [TL;DR](#tldr)
- [In one paragraph](#in-one-paragraph)
- [The result, up front](#the-result-up-front)
- [What it actually does](#what-it-actually-does)
- [How it works — the pipeline](#how-it-works--the-pipeline)
- [The core mechanism, in one real example](#The-core-mechanism-in-one-real-example)
- [Evaluation integrity](#evaluation-integrity)
- [On the explanation layer, and the template fallback](#on-the-explanation-layer-and-the-template-fallback)
- [Feature list](#feature-list)
- [What makes this different](#what-makes-this-different)
- [This is an MVP](#this-is-an-mvp)
- [Honest scope and limitations](#honest-scope-and-limitations)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Running locally](#running-locally)
- [Testing](#testing)
- [Credits](#credits)

## TL;DR

- Recoverly turns a failed payment into a decision: **why it failed,
  what to do, when, and why** — not a blind retry.
- On a frozen held-out test set, it recovers **₹660,008.25** more than
  a standard fixed-delay baseline — **36.4% vs. 28.2%** recovery rate.
- The comparison is policy-blind and reproducible by construction, not
  just favorable .
- This is a deliberately scoped MVP: the decision engine is built
  and verified first; the operational layer around it is the
  documented next step.

---

## In one paragraph

Most payment recovery systems treat every failed transaction the same
way: wait a fixed delay, retry, hope. Recoverly classifies why a
payment failed, scores what to do about it by expected recovered
value — not just probability — and explains the decision in
plain language, grounded strictly in its own numbers. On a frozen,
held-out test set, this reason-aware approach recovers **₹660,008.25**
more than a standard fixed-delay retry baseline, measured against an
evaluation method designed specifically so that number can't be
accidentally rigged in its own favor. Full reasoning below.

---

## The result, up front

On a frozen, held-out set of failed transactions the model never saw
during training or tuning:

| Policy | Recovery rate | ₹ recovered |
|---|---|---|
| Baseline (fixed-delay retry, 4h, max 2 attempts) | 28.2% | ₹3,658,072.15 |
| Recoverly (reason-aware, timing-optimized) | 36.4% | ₹4,318,080.40 |
| **Incremental uplift** | — | **₹660,008.25** |

Both figures are computed on the exact same held-out transactions,
using the exact same outcome model, so the comparison is apples to
apples.

---

## What it actually does

For every failed payment, Recoverly answers five questions in
sequence:

1. **What happened?** — classifies the failure reason (insufficient
   funds, bank timeout, expired card, gateway error, issuer decline,
   or 3DS drop-off).
2. **Is it worth recovering?** — estimates the expected recovered ₹
   value of each possible response, not a bare probability.
3. **What should we do?** — selects the highest-expected-value action
   from a fixed, safe set: retry now, retry at a scheduled window,
   request a card update, escalate to manual review, or take no
   action.
4. **Why?** — produces a plain-language explanation, grounded strictly
   in the numbers behind the decision. It cannot state a figure that
   isn't already part of the decision itself — enforced by an
   automated check, not just a design intention.
5. **Did it work?** — every decision is evaluated against a fair,
   held-out baseline, so impact is measured, not asserted.

---

## How it works — the pipeline

```
Failed transaction
        ↓
Feature engineering (leakage-safe: amount, hour of day, day of month,
card type, recurring flag, customer past-failure count and reason
distribution, attempt number — never the failure reason itself)
        ↓
Classifier → predicted failure reason + confidence
        ↓
Confidence < 55%? → escalate to manual review, no guessing
        ↓
Policy engine → scores every candidate action/timing window by its
own estimated expected recovered value, selects the best
        ↓
Explanation layer → plain-language justification, grounded strictly
in the decision's own numbers
        ↓
API response → reason, action, retry timing, expected value,
explanation, and the full set of scored candidates
        ↓
Dashboard → live decisioning and held-out evaluation, side by side
```

### The core mechanism, in one real example

For an `insufficient_funds` failure (shown in the dashboard as demo
case `retry_scheduled`), instead of retrying blindly, the policy
scores every timing candidate:

| Action | Retry at | Expected recovered value |
|---|---|---|
| Retry now | immediately | ₹80 |
| Retry scheduled | +4 hours | ₹140 |
| Retry scheduled | +24 hours | ₹200 |
| Retry scheduled | next likely salary date, 09:00 local | **₹820** |

Waiting for a plausible salary-credit window recovers roughly 5× more
expected value than retrying immediately. This timing-awareness over
blind repetition is the entire technical contribution of this
project — and the held-out evaluation above is what proves it
generalizes across the full test set, not just this one example.

A second illustrative case, `notify_update_card`, shows the same
policy correctly refusing to retry an `expired_card` failure at all —
retrying a card that can't currently be charged wastes an attempt for
zero expected value, so the system routes to a card-update request
instead. This safety boundary is enforced in code, not left to the
model's judgment.

---

## Evaluation integrity

A ₹-uplift number is only meaningful if it's measured fairly. The
following are what make this comparison fair, not favorable:

- **Policy-blind ground truth.** Whether a retry actually succeeds is
  decided by a separate outcome model that has no knowledge of which
  policy — baseline or Recoverly — chose the timing. It only ever
  receives a transaction and a proposed retry time. Neither policy can
  see or influence this model's internal logic; enforced by an
  automated check that the policy code has zero import dependency on
  it.
- **Deterministic, reproducible outcomes.** Given the same transaction
  and retry time, the outcome model always returns the same result —
  re-running the evaluation produces byte-identical figures, verified
  by test.
- **Identical held-out transactions for both policies.** The test set
  is frozen and checksum-verified after being split; any modification
  to it would fail an automated integrity check.
- **A real, unflattering baseline.** Baseline is a standard
  fixed-delay retry (4 hours, max 2 attempts) — the kind of policy
  most systems actually run today, not a strawman built to lose.
- **Confidence-gated safety.** Below 55% classification confidence,
  the system escalates to manual review instead of guessing — verified
  by test across values above, at, and below the threshold.
- **Leakage-safe features.** The classifier's inputs never include the
  failure reason, the success outcome, or any retry-outcome field —
  only information genuinely available before a decision is made;
  enforced by an automated leakage check, not a comment.
- **Disjoint demo and evaluation data.** The illustrative demo cases
  shown interactively in the dashboard (`retry_scheduled`,
  `notify_update_card`, `retry_now`, `no_action`,
  `escalate_manual_review`) are sourced from the training/development
  split and have zero transaction-ID overlap with the held-out test
  set used for the reported metrics above — verified by test.

Metric definitions, exactly:
```
recovery_rate      = recovered_count / total_failed_attempts     (per policy)
₹_recovered        = sum(recovered_amount)                        (per policy)
incremental_uplift = ₹_recovered[Recoverly] − ₹_recovered[baseline]
                      (same held-out transactions, same outcome model, both policies)
```

---

## On the explanation layer, and the template fallback

Every decision includes a plain-language explanation, produced one of
two ways:

- **Live path:** a generative model (Gemini) converts the decision's
  structured fields into natural language. Every number the model
  states is checked against the decision's actual values before being
  shown; output that introduces an ungrounded figure is rejected, with
  one retry allowed before falling back.
- **Fallback path:** a deterministic template, used whenever the live
  path isn't available — no API key configured, a timeout, or a failed
  grounding check.

**In this deployment and demo video, the fallback path is what you'll
see**, because no live API key is configured in this environment. This
is intentional, visible, tested behavior — not a failure being hidden.
A system that silently breaks or hallucinates under a missing
dependency is worse than one that degrades gracefully to a
deterministic, still-accurate explanation. That fallback behavior was
itself deliberately engineered and tested, not left as an afterthought.

---

## Feature list

**Core decisioning engine**
- Synthetic transaction generator with documented, reproducible
  failure-reason correlations (seeded, deterministic regeneration)
- Policy-blind outcome model — the sole source of ground truth for
  retry success, fully isolated from both policies
- Failure-reason classifier (gradient-boosted trees) with leakage-safe
  features, evaluated once on a frozen, checksum-verified held-out
  split
- Confidence-based routing — low-confidence predictions never guess
- Baseline and smart retry-policy engines sharing one contract, with
  hard safety boundaries (e.g. an expired card is never blindly
  retried, max two attempts per transaction, no duplicate scheduling)
- Grounded explanation layer with a tested, deterministic fallback

**Evaluation and evidence**
- Held-out policy comparison — recovery rate, ₹ recovered, incremental
  uplift, computed identically for both policies on the same
  transactions
- Recovery opportunity matrix — predicted opportunities by transaction
  size and policy-belief recovery probability
- Revenue-at-risk breakdown by failure reason, from observed held-out
  data
- Model transparency — real feature importances from the trained
  classifier

**Interaction**
- Illustrative demo cases spanning all five possible actions, sourced
  separately from the held-out evaluation set
- Scored policy candidates displayed alongside every decision — not
  just the winning action, but every option the policy considered and
  why it lost
- Live recovery simulator — any scenario submitted runs the full live
  pipeline in real time
- Decision trace — a per-transaction visual walkthrough of
  Classify → Decide → Explain
- AUTO / REVIEW / BLOCKED status badges, mapped from each decision's
  action — a presentation-only label, never a second decision layer
- Predicted-vs-Actual labeling and color coding throughout, so a
  viewer can distinguish a policy's belief from a verified outcome at
  a glance

---

## What makes this different

Most "AI revenue recovery" submissions in a track like this take one
of two shortcuts: an LLM that decides everything end-to-end with no
verifiable evaluation, or a prediction model with no baseline to prove
it actually beats doing nothing clever. Recoverly is built to avoid
both:

| | Typical LLM-wrapper approach | Recoverly |
|---|---|---|
| Who decides the action | The LLM, end to end | A deterministic policy engine — the LLM only narrates a decision already made |
| Can the explanation invent numbers | Usually unchecked | No — every stated figure is checked against the decision's actual values |
| Is the uplift number defensible | Often asserted, rarely tested | Policy-blind, reproducible, checksum-verified held-out comparison |
| What happens on low confidence | Often nothing — the model just answers | Escalates to manual review, verified by test |
| What happens if the LLM API fails | Often breaks or hallucinates | Falls back to a deterministic, still-accurate template |

None of this is a claim of superior intelligence — it's a claim of
verifiability. Every number in this README is something you can
re-run and check, not something you have to take on faith.

---

## This is an MVP

Recoverly was deliberately scoped to prove the decisioning engine was
correct — leak-free features, a policy-blind evaluation, a tested
explanation layer — before building the operational tooling around it.
That ordering was a choice: an operationally rich dashboard sitting on
top of an unverified decision engine would be a weaker project than a
rigorously verified engine with a simpler dashboard.

The architecture already supports what comes next:

- **Recovery command center** — batch review of high-value cases, with
  human-in-the-loop approval for low-confidence, high-value decisions
- **Closed-loop outcome tracking** — every real action's actual result
  feeding back into a structured outcome log
- **Periodic retraining** — using that logged feedback to retrain the
  classifier and refine the policy's belief table on real, not
  synthetic, outcomes
- **Multi-channel recovery messaging** — personalized customer
  communication (email, SMS, WhatsApp), built on the same
  grounded-explanation approach already used for internal decisions
- **Learned retry-timing policy** — replacing the current hand-coded
  belief table with a model that learns optimal timing directly from
  accumulated real outcomes, rather than fixed candidate windows

---

## Honest scope and limitations

- All results in this README and demo are computed on a carefully
  engineered synthetic dataset, with fully documented and
  inspectable generation rules — not a production validation on real
  transaction data.
- The contribution being demonstrated is the mechanism: that
  reason-aware, timing-optimized decisions recover more expected value
  than naive fixed-delay retries, measured against a fair baseline.
  That mechanism is what should carry over to real data — the exact ₹
  figures above will not.
- The policy's timing-candidate probabilities come from a hand-coded
  belief table, not a learned model — a deliberate, auditable choice,
  at the cost of not adapting automatically to new patterns. Listed
  above as a roadmap item, not hidden as a limitation.
- Retry-timing candidates are described as "optimized among modeled
  windows," not as "the best possible time" — the system selects the
  best of a defined, documented action space, and no claim beyond that
  is made.
- The live explanation path requires a configured API key; this
  deployment intentionally runs on its fallback path.

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend / API | Python, FastAPI |
| Classifier | scikit-learn, gradient-boosted trees |
| Explanation | Gemini API, with a deterministic template fallback |
| Dashboard | Streamlit |
| Tests | pytest |

---

## Project structure

```
recoverly/
├── .gitignore
├── pyproject.toml
├── README.md
│
├── backend/
│   ├── __init__.py
│   ├── classifier.py             # failure-reason classifier
│   ├── evaluation.py             # held-out policy comparison
│   ├── explain.py                # grounded explanation + fallback
│   ├── features.py               # leakage-safe feature engineering
│   ├── policy_belief_table.py    # policy's own probability estimates
│   ├── retry_policy.py           # baseline + smart policy engines
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI routes
│   │   └── schemas.py            # request/response schemas
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py          # synthetic transaction generator
│   │   ├── outcome_model.py      # policy-blind ground truth
│   │   ├── transactions.csv
│   │   ├── train_set_v1.csv
│   │   ├── dev_set_v1.csv
│   │   ├── test_set_v1.csv
│   │   ├── retry_outcomes.csv
│   │   ├── demo_cases.csv
│   │   └── explanation_cache.json
│   │
│   ├── models/                   # trained classifier, feature importances
│   │
│   └── dashboard/
│       ├── app.py                # Streamlit dashboard
│       └── styles.css            # theme tokens and custom styling
│
└── tests/                        # full pytest suite, one file per module
    ├── test_api_integration.py
    ├── test_classifier.py
    ├── test_confidence_routing.py
    ├── test_dashboard_data.py
    ├── test_evaluation.py
    ├── test_explain.py
    ├── test_features_and_splits.py
    ├── test_generator.py
    ├── test_outcome_model.py
    ├── test_project_smoke.py
    └── test_retry_policy.py
```

## Running locally

```bash
# install dependencies
pip install -r requirements.txt

# regenerate the synthetic dataset (deterministic, seeded)
python backend/data/generator.py

# train the classifier
python backend/classifier.py

# run the API
uvicorn backend.api.main:app --reload

# run the dashboard
streamlit run backend/dashboard/app.py
```

To enable the live explanation path instead of the template fallback:

```bash
$env:GEMINI_API_KEY="your_key_here"
```

---

## Testing

Every module is tested independently, and the tests exist specifically
to enforce the evaluation-integrity guarantees stated above, not just
to check that the code runs:

- Deterministic data generation and reproducible outcome sampling
- Outcome-model policy-blindness and salary-window/expired-card
  behavior
- Leakage-safe features, customer-disjoint splits, held-out checksum
  integrity
- Classifier accuracy (macro-F1) on the frozen test set
- Retry-policy safety boundaries and zero import dependency on the
  outcome model
- Confidence-based routing at, above, and below the 55% threshold
- Held-out evaluation correctness — identical transactions, identical
  outcome model, for both policies
- Explanation grounding, including a deliberately adversarial
  invented-number case
- Demo-case and held-out test-set disjointness
- Full API integration, end to end

```bash
pytest
```

---

## Credits

Built solo by Pranav for the Razorpay Buildathon.
