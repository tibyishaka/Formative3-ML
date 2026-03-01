# Formative 3 — Machine Learning & Probability

This repository contains the implementation and analysis for Formative Assessment 3, covering probability distributions, Naive Bayes sentiment analysis, and gradient descent (both manual and in code).

---

## Table of Contents

1. [Part 1: Bivariate Normal Distribution](#part-1-bivariate-normal-distribution)
2. [Part 2: Naive Bayes — Sentiment Analysis](#part-2-naive-bayes--sentiment-analysis)
3. [Part 3: Manual Gradient Descent](#part-3-manual-gradient-descent)
4. [Part 4: Gradient Descent in Code](#part-4-gradient-descent-in-code)
5. [Datasets](#datasets)
6. [Setup & Usage](#setup--usage)
7. [Repository Structure](#repository-structure)

---

## Part 1: Bivariate Normal Distribution

**Notebook:** `Probability-Distributions.ipynb`

**Objective:** Compute probability density values for a dataset using the bivariate normal distribution formula, implemented entirely from scratch (no statistical libraries).

**Dataset:** [`Global_Education.csv`](https://www.kaggle.com/code/nelgiriyewithana/introduction-to-world-educational-data/input?select=Global_Education.csv) — a global education statistics dataset.

**Features selected:**
- `Gross_Tertiary_Education_Enrollment`
- `Gross_Primary_Education_Enrollment`

**What was done:**
1. Loaded and explored the dataset (`.info()`, `.isnull().sum()`, `.describe()`).
2. Selected the two continuous features above.
3. Computed the **mean vector** using NumPy.
4. Computed the **covariance matrix** manually (centered data approach, no `np.cov`).
5. Implemented the **bivariate normal PDF** from scratch using NumPy determinant and inverse.
6. Computed PDF values for every data point (NumPy only — no `scipy.stats`).
7. Created a visualization grid and produced two plots:
   - A **contour plot** (density levels over the 2D feature space, with data points overlaid).
   - A **3D surface plot** of the PDF.

---

## Part 2: Naive Bayes — Sentiment Analysis

**Notebook:** `bayes_imdb_dataset.ipynb`

**Objective:** Apply Bayes' Theorem to compute posterior probabilities for sentiment keywords in movie reviews.

**Dataset:** [IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (50 000 reviews, loaded with `pandas`).

### Preprocessing

- Converted all reviews to lowercase.
- Removed HTML line breaks (`<br />`).
- Stripped all punctuation.

### Keyword Selection

| Sentiment | Keywords |
|-----------|----------|
| Positive  | `superb`, `wonderful`, `excellent`, `fantastic` |

### Conditional Probability Direction

Computed **P(Positive | keyword)** for each selected keyword.

### Bayes' Theorem Applied

P(Positive | keyword) = [ P(keyword | Positive) * P(Positive) ] / P(keyword)

### What was computed per keyword

| Probability | Description |
|-------------|-------------|
| Prior P(Pos) | Fraction of positive reviews in the full dataset |
| Likelihood P(kw \| Pos) | Fraction of positive reviews containing the keyword |
| Marginal P(kw) | Fraction of all reviews containing the keyword |
| Posterior P(Pos \| kw) | Probability a review is positive given it contains the keyword |

### Implementation Notes

- Pure Python + `pandas` only — no `sklearn`, `nltk`, or ML libraries.
- All counting and probability steps are explicit and manual.

---

## Part 3: Manual Gradient Descent

**Files:** `gradient-descent-calculations/` (one PDF per group member)

**Objective:** Manually work through four iterations of gradient descent for a simple linear regression model, showing every intermediate calculation.

### Setup

Linear model: y_hat = m*x + b

| Parameter | Value |
|-----------|-------|
| Initial m | -1 |
| Initial b | 1 |
| Learning rate (alpha) | 0.1 |
| Data points | (1, 3) and (3, 6) |
| Iterations | 4 (one per group member) |

### Update Rules

- Gradient w.r.t. m: (-2/n) * sum( x_i * (y_i - y_hat_i) )
- Gradient w.r.t. b: (-2/n) * sum( y_i - y_hat_i )
- m <- m - alpha * (dJ/dm)
- b <- b - alpha * (dJ/db)

Each group member computed one iteration by hand; their working is saved as a PDF inside `gradient-descent-calculations/`.

---

## Part 4: Gradient Descent in Code

**Notebook:** `Gradient_Descent_Manual_Calculation.ipynb`

**Objective:** Translate the manual gradient descent process into Python, printing every intermediate step and plotting convergence.

### Setup (matches Part 3)

| Parameter | Value |
|-----------|-------|
| Initial m | -1 |
| Initial b | 1 |
| Learning rate (alpha) | 0.1 |
| Data points | (1, 3) and (3, 6) |
| Iterations | 4 |

### What was implemented

- Explicit iterative updates of m and b (no black-box optimisers).
- A printed table at each iteration showing: m, b, MSE, dJ/dm, dJ/db.
- Two **Matplotlib** plots:
  1. **Parameter Updates** — m and b vs. iteration number.
  2. **Error Reduction** — MSE vs. iteration number.

---

## Datasets

| File | Description |
|------|-------------|
| `Global_Education.csv` | Global education statistics — used for the bivariate normal distribution (Part 1) |
| IMDb Movie Reviews | External Kaggle dataset — used for Naive Bayes sentiment analysis (Part 2) |

---

## Setup & Usage

### Requirements

```bash
pip install numpy pandas matplotlib jupyter
```

### Running the Notebooks

```bash
jupyter notebook Probability-Distributions.ipynb
jupyter notebook bayes_imdb_dataset.ipynb
jupyter notebook Gradient_Descent_Manual_Calculation.ipynb
```

> **Note:** `bayes_imdb_dataset.ipynb` expects the IMDb CSV at `/content/IMDB Dataset.csv` (Google Colab path). Update the path if running locally.

---

## Repository Structure

```
Formative3-ML/
├── Probability-Distributions.ipynb           # Part 1 — Bivariate Normal Distribution
├── bayes_imdb_dataset.ipynb                  # Part 2 — Naive Bayes Sentiment Analysis
├── Gradient_Descent_Manual_Calculation.ipynb # Part 4 — Gradient Descent in Code
├── gradient-descent-calculations/            # Part 3 — Hand-written iteration PDFs
│   ├── First iteration.pdf
│   ├── Christian Tenny Gentel Iradukonda.pdf
│   ├── Kabasinga Arsene.pdf
│   └── IBYISHAKA-Last-Iteration.PDF
├── Global_Education.csv                      # Dataset for Part 1
└── README.md
```
