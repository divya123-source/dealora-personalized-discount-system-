# DealoRa: Personalized Discount Recommendation System

DealoRa is a Flask web application where customers browse products, get personalised recommendations and discount suggestions, and place orders. Admins manage products, discounts, customers and orders, and can review the results of the recommendation model experiment on a dashboard.

> **Honest note on results:** the 85-90% target was a goal, not a promise. The measured results are reported exactly as they came out of the evaluation (see [Model evaluation](#model-evaluation)). The target was **not met**.

## Features

- Customer accounts with login (JWT), cart, wishlist, orders and notifications
- Product catalogue with categories and brand details
- Discounts (percentage or fixed, coupon codes, minimum purchase, validity dates) and redemption history
- Personalised recommendations on the **live DealoRa catalog** with a short reason for each product and a suggested discount
- Customer segmentation and price-sensitivity signals used for discount suggestions
- Popularity fallback when the model fails or has nothing to rank (new users, empty history)
- Admin area: overview, products, discounts, customers, orders, redemptions and a **Model evaluation** section

## Project structure

```
backend/
  app.py                    Flask app factory and entry point
  config.py                 Configuration
  models.py                 Database models (User, Product, Discount, Order, ...)
  seed.py                   Demo data (needs passwords from environment variables)
  reset_admin_password.py   Set a new password for an existing user
  routes/                   API blueprints (auth, products, discounts, orders,
                            recommendations, notifications, admin, model_evaluation)
  ml/                       Recommendation, segmentation, price sensitivity,
                            discount optimisation
frontend/                   HTML pages, css/ and js/ (served by Flask)
data/
  raw/                      Original datasets (kept untouched)
  processed/                Cleaned data and trial results (model_card.json,
                            retailrocket_experiment.json)
tests/
requirements.txt
.env.example
```

## Setup (Windows PowerShell)

1. Clone the repository and open the folder in Visual Studio Code.
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   On macOS or Linux: `python3 -m venv .venv` and `source .venv/bin/activate`.
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in the values. Never commit `.env`.
5. Run the app:
   ```
   python backend\app.py
   ```
6. Open `http://127.0.0.1:5000` in the browser. The admin page is `http://127.0.0.1:5000/admin.html`.

## Demo data and admin login

Demo users are created by `backend/seed.py`. Passwords are **not** stored in the code. Set a unique password of at least 12 characters for each demo user as an environment variable, then run the seed:

```
$env:DEALORA_DEMO_ADMIN_PASSWORD = "choose-a-password"
$env:DEALORA_DEMO_CUSTOMER_PASSWORD = "choose-a-password"
$env:DEALORA_DEMO_MAYA_PASSWORD = "choose-a-password"
$env:DEALORA_DEMO_ETHAN_PASSWORD = "choose-a-password"
$env:DEALORA_DEMO_SOFIA_PASSWORD = "choose-a-password"
python backend\seed.py
```

The seed adds categories, products, discounts and users only if they do not exist yet. It does not delete existing data. Admin account: `admin@dealora.com`; customer account: `user@dealora.com`.

To set a new password for an existing user (for example if the admin password is forgotten):

```
$env:DEALORA_NEW_PASSWORD = "choose-a-new-password"
python backend\reset_admin_password.py
```

Set `DEALORA_RESET_EMAIL` to reset a different user. Back up `backend\dealora.db` first.

## API overview

| Endpoint | Access | Purpose |
|---|---|---|
| `GET /api/recommendations?limit=8` | Logged-in user | Ranked products with `recommendation_reason`, `suggested_discount`, `source` and `fallback_used` |
| `GET /api/model/evaluation` | Public (read-only) | Real evaluation results read from `data/processed/trial/*.json` |
| `/api/auth`, `/api/products`, `/api/discounts`, `/api/orders`, `/api/notifications` | Varies | Core store features |
| `/api/admin/*` | Admin | Admin overview, products, discounts, users, orders, redemptions |

The recommendation response states `"offline_retailrocket_ids_used": false`: website recommendations are computed from the DealoRa catalog, not from the offline experiment.

## Model evaluation

The recommendation experiment used the public **Retailrocket** e-commerce events dataset (2,755,641 valid interactions, 0 rejected rows). Its item IDs are not DealoRa product IDs, so this model is **not used in the live store**. The results are shown on the admin dashboard as offline evaluation only.

**Protocol**

- Global chronological 80/20 holdout; test timestamps are never used for training, validation or tuning
- Two walk-forward cross-validation folds on the training period (50-65% and 65-80%)
- Candidates: popularity baseline and three co-visitation models; selection metric: mean NDCG@10 across folds
- Selected model: `covisitation_w10_n100` (100 neighbours, recent window 10)

**Final test results (1,990 evaluation users)**

| Metric | Popularity baseline | Selected model |
|---|---|---|
| NDCG@10 | 0.0111 | 0.0151 |
| HitRate@10 | 0.0186 | 0.0246 |
| MAP@10 | 0.0094 | 0.0124 |
| Recall@10 | 0.0121 | 0.0191 |
| Precision@10 | 0.0019 | 0.0029 |

The selected model beats the popularity baseline on every ranking metric, but the absolute scores are low and the 85-90% target was not met.

**Limitations**

- Retailrocket purchases are sparse relative to the item catalogue, so Precision@K is naturally low; HitRate@K and NDCG@K are the more informative measures
- Only two cross-validation folds were used
- Classification-style metrics on sampled candidates (ROC-AUC about 0.51) are close to chance and should not be read as catalogue-wide accuracy
- Seed data in the live store is tiny (a handful of users and orders), so live recommendations are a demonstration rather than a validated system

**Possible improvements:** session-based sequential models (for example GRU4Rec or SASRec) on a GPU, item content features if a verified mapping to DealoRa products exists, and larger evaluation samples.

## Troubleshooting

- **`ImportError: DLL load failed ... Application Control policy has blocked this file` (Windows):** the pre-compiled SQLAlchemy extension is blocked. Reinstall SQLAlchemy without the compiled extension:
  ```
  pip uninstall -y sqlalchemy
  $env:DISABLE_SQLALCHEMY_CEXT = "1"
  pip install --no-binary sqlalchemy --no-cache-dir sqlalchemy
  ```
- **Admin page says "This space is for admins":** sign in with an account whose role is `admin`.
- **Model evaluation section says results are unavailable:** check that `data/processed/trial/model_card.json` and `retailrocket_experiment.json` exist and that the server is running.
- **Page not found for `css/model_evaluation.css`:** the file must be in `frontend/css/`, next to `style.css`.

## Safety rules for contributors

- Do not commit passwords, API keys, `.env` or the database files (`dealora.db`, `dealora.db.bak`)
- Do not upload large raw datasets unless necessary
- Create a branch for new work and write meaningful commit messages
- Do not force-push over another person's work
- Every metric shown on the dashboard must come from a real evaluation
