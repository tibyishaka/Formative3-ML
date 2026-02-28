# Formative 3 — Machine Learning & Probability

This repository contains the implementation and analysis for Formative Assessment 3, covering probability distributions, Naive Bayes sentiment analysis, and gradient descent (both manual and in code).

---

## Table of Contents

1. [Part 1: Bivariate Normal Distribution](#part-1-bivariate-normal-distribution)
2. [Part 2: Naive Bayes — Sentiment Analysis](#part-2-naive-bayes--sentiment-analysis)
3. [Part 3: Manual Gradient Descent](#part-3-manual-gradient-descent)
4. [Part 4: Gradient Descent in Code](#part-4-gradient-descent-in-code)
5. [Dataset](#dataset)
6. [Setup & Usage](#setup--usage)

---

## Part 1: Bivariate Normal Distribution

**Objective:** Compute probability density values for a dataset using the bivariate normal distribution formula, implemented entirely from scratch (no statistical libraries).

**Dataset:** A relevant dataset sourced online (see `Global_Education.csv`).

**What was done:**
- Selected two continuous variables from the dataset to form a bivariate distribution.
- Manually implemented the bivariate normal PDF formula:

$$f(x, y) = \frac{1}{2\pi\sigma_x\sigma_y\sqrt{1-\rho^2}} \exp\!\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_x)^2}{\sigma_x^2} - \frac{2\rho(x-\mu_x)(y-\mu_y)}{\sigma_x\sigma_y} + \frac{(y-\mu_y)^2}{\sigma_y^2}\right]\right)$$

- Computed PDF values for each data point using only NumPy (no `scipy.stats` or similar).
- Visualized the distribution using **Matplotlib**:
  - A **contour plot** showing density levels across the 2D feature space.
  - A **3D surface plot** of the PDF.

---

## Part 2: Naive Bayes — Sentiment Analysis

**Objective:** Apply Bayes' Theorem to compute posterior probabilities for sentiment keywords in movie reviews.

**Dataset:** [IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), loaded using `pandas`.

### Keyword Selection

| Sentiment  | Keywords Chosen               |
|------------|-------------------------------|
| Positive   | 'superb', 'wonderful', 'excellent', 'fantastic' |


### Conditional Probability Direction

The group computed **P(Positive | keyword)** for each selected keyword.

### Probability Table

For each keyword, the following probabilities were computed:

| Term        | Prior P(Pos) | Likelihood P(kw\|Pos) | Marginal P(kw) | Posterior P(Pos\|kw) |
|-------------|-------------|----------------------|----------------|----------------------|
| superb      | ...         | ...                  | ...            | ...                  |
| Wonderful  | ...         | ...                  | ...            | ...                  |
| excellent        | ...         | ...                  | ...            | ...                  |
| Fantastic    | ...         | ...                  | ...            | ...                  |


### Implementation Notes

- Bayes' Theorem applied:

$$P(\text{Positive} \mid \text{keyword}) = \frac{P(\text{keyword} \mid \text{Positive}) \cdot P(\text{Positive})}{P(\text{keyword})}$$

- Implemented using basic Python only — no `sklearn`, `nltk`, or ML libraries.
- All counting and probability calculations are done manually with standard Python operations.

---

## Part 3: Manual Gradient Descent

**Objective:** Manually compute three iterations of gradient descent for a simple linear regression model, showing every calculation step.

### Setup

Given the linear model:

$$\hat{y} = mx + b$$

- Initial $m = 0$, $b = 0$
- Learning rate $\alpha = 0.01$
- Data points: $(1, 2)$ and $(3, 4)$

### Process

For each iteration:
1. **Compute predictions** $\hat{y}_i$ using current $m$ and $b$.
2. **Derive the MSE cost function gradients:**

$$\frac{\partial J}{\partial m} = -\frac{2}{n}\sum_{i=1}^{n} x_i(y_i - \hat{y}_i)$$

$$\frac{\partial J}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)$$

3. **Update parameters:**

$$m \leftarrow m - \alpha \cdot \frac{\partial J}{\partial m}$$

$$b \leftarrow b - \alpha \cdot \frac{\partial J}{\partial b}$$

The number of iterations performed equals the number of group members. All intermediate results are shown in the submitted handwritten/typed document.

### Trend Observation

After each iteration, both $m$ and $b$ increase toward values that minimize the error — confirming that gradient descent is converging and reducing the MSE cost with each update.

---

## Part 4: Gradient Descent in Code

**Objective:** Translate the manual gradient descent process into Python code using `SciPy` where appropriate, with each update step clearly visible.

**What was implemented:**
- Iterative updates of $m$ and $b$ over multiple iterations (no black-box abstractions).
- Prediction of $\hat{y}$ using the final learned $m$ and $b$.
- Two separate **Matplotlib** plots:
  1. **m and b vs. Iterations** — showing how the parameters evolve.
  2. **MSE Error vs. Iterations** — showing convergence of the cost function.

---

## Dataset

| File | Description |
|------|-------------|
| `Global_Education.csv` | Used for the bivariate normal distribution (Part 1) |
| IMDb Movie Reviews | External dataset for Naive Bayes sentiment analysis (Part 2) |

---

## Setup & Usage

### Requirements

```bash
pip install numpy pandas matplotlib scipy jupyter
```

### Running the Notebook

```bash
jupyter notebook Probability-Distributions.ipynb
```

All parts of the assignment are implemented inside `Probability-Distributions.ipynb`.

---

## Repository Structure

```
Formative3-ML/
├── bayes_imdb_dataset.ipynb
├── Probability-Distributions.ipynb
├── calculations/
│   ├── iteration-1-[colleague].pdf
│   └── iteration-2-tonny.pdf
├── Global_Education.csv
└── README.md
```
