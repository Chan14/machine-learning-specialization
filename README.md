# ml-specialization-playground

My work through the DeepLearning.AI / Stanford **Machine Learning
Specialization**: the course labs, my re-implementations, my own experiments,
and the reusable code that came out of them.

## Table of contents

- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Everyday commands](#everyday-commands)
- [Where new work goes](#where-new-work-goes)
- [Course 1: Supervised Machine Learning](#course-1-supervised-machine-learning)
- [Course 2: Advanced Learning Algorithms](#course-2-advanced-learning-algorithms)
- [Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning](#course-3-unsupervised-learning-recommenders-reinforcement-learning)
- [Reusable package: `mlspec`](#reusable-package-mlspec)
- [Books](#books)
- [Scratch](#scratch)

## Repository layout

```
ml-specialization-playground/
├── pyproject.toml          # package metadata, dependencies, pytest and ruff config
├── requirements.txt        # exact pinned snapshot of ml-spec-env
├── .python-version         # pinned interpreter (3.12)
├── course1_supervised_learning/
│   ├── certificate/
│   └── w1_.../ w2_.../ w3_.../        # one folder per week
├── course2_advanced_learning_algorithms/
├── course3_unsupervised_learning_recommenders_rl/
├── src/mlspec/             # my reusable, tested code
├── tests/                  # mirrors src/mlspec/
├── books/                  # exercises from books outside the course
└── scratch/                # rough work that isn't tied to a course week
```

**Inside each week, the same folder names mean the same thing:**

| Folder | What's in it |
| ------ | ------------ |
| `optional_labs/` | Course-provided labs (solutions), with their `lab_utils*.py`, images, and data |
| `labs_practice/` | My own re-implementations of those labs |
| `programming_assignments/aN_<name>/` | Graded assignments, each with its own utilities and tests |
| `practice_quizzes/pN_<topic>/` | Screenshots of practice quizzes |
| `my_code/` | My own scripts and mini-projects for that week |
| `archive*/` | Older versions kept for reference |

**Rules that keep this layout sound**

1. Course lab folders are self-contained. A notebook imports its `lab_utils*.py`
   from its own folder and reads `images/` and `data/` next to it. Move a lab
   folder only as a whole.
2. Code worth importing from more than one place goes in `src/mlspec/`, with tests
   in the mirrored `tests/` path. Notebooks import it as
   `from mlspec.<module> import ...`, and never use `sys.path` hacks.
3. Scripts find their data with `Path(__file__).resolve().parent / "data"`, so
   they work from any working directory.
4. Any folder named `untracked/`, at any depth, is ignored by git.

## Setup

The environment is the pyenv virtualenv `ml-spec-env` (Python 3.12). TensorFlow
is heavily used here, so this repo stays on 3.12.

One time, from the repo root:

```bash
brew install graphviz              # system tool `dot`, used by the decision tree plots
pyenv activate ml-spec-env
pip install -e ".[dev]"            # editable install of mlspec + dev tools
nbstripout --install               # strip notebook outputs on commit; local files keep them
pytest                             # expect all tests to pass
```

To rebuild the exact environment on a new machine:
`pip install -r requirements.txt && pip install -e . --no-deps`.
After changing the environment, refresh the snapshot with
`pip freeze --exclude-editable > requirements.txt`.

## Everyday commands

```bash
pyenv activate ml-spec-env                     # once per terminal
pytest                                         # run all tests
python course1_supervised_learning/w1_univariate_linear_regression/my_code/learning_rate_experiments.py
ruff check src tests                           # lint my package
```

## Where new work goes

| You are writing...                            | Put it in                                  |
| --------------------------------------------- | ------------------------------------------ |
| Course lab files (downloaded)                 | `courseN_.../wN_.../optional_labs/`        |
| Your re-implementation of a lab               | `courseN_.../wN_.../labs_practice/`        |
| Your own script or mini-project for a week    | `courseN_.../wN_.../my_code/`              |
| A function you'll reuse across notebooks      | `src/mlspec/...` + `tests/...`             |
| Rough work for a specific course              | `courseN_.../scratch/`                     |
| Rough work not tied to a course               | `scratch/`                                 |
| Anything that must never reach GitHub         | any `untracked/` folder                    |

## Course 1: Supervised Machine Learning

Certificate: [`certificate/course1_certificate.pdf`](course1_supervised_learning/certificate/course1_certificate.pdf)

### [Week 1: Univariate linear regression](course1_supervised_learning/w1_univariate_linear_regression/)

- **Optional labs:** Jupyter intro, model representation, cost function, gradient descent ([`optional_labs/`](course1_supervised_learning/w1_univariate_linear_regression/optional_labs/); `archive_beta_version/` has the older beta release)
- **My practice:** labs 2 to 4 ([`labs_practice/`](course1_supervised_learning/w1_univariate_linear_regression/labs_practice/))
- **My code:** [`learning_rate_experiments.py`](course1_supervised_learning/w1_univariate_linear_regression/my_code/learning_rate_experiments.py) compares learning rates, early stopping, and divergence on synthetic data, with plots from [`plotters.py`](course1_supervised_learning/w1_univariate_linear_regression/my_code/plotters.py). The model itself is [`mlspec.linear_regression.univariate`](src/mlspec/linear_regression/univariate.py).
- **Practice quizzes:** supervised vs unsupervised, regression, gradient descent

### [Week 2: Multivariate linear regression](course1_supervised_learning/w2_multivariate_linear_regression/)

- **Optional labs:** NumPy vectorization, multiple variables, feature scaling and learning rate, feature engineering and polynomial regression, scikit-learn (GD and normal equation)
- **My practice:** labs 1 to 5, plus [`lr_with_alex.ipynb`](course1_supervised_learning/w2_multivariate_linear_regression/labs_practice/lr_with_alex.ipynb), a scikit-learn linear regression mini-project
- **My code** ([`my_code/`](course1_supervised_learning/w2_multivariate_linear_regression/my_code/)):
  - `simple_mv_lin_regression.py`: loop vs vectorized prediction, cost, and gradient
  - `feature_scaling_learning_curve.py`: feature scaling and learning curves on the housing data
  - `feature_eng_and_poly_reg.py`: feature engineering and polynomial regression
  - `lin_regression_lr_scikit_learn.py`, `lin_regression_sgrd_scikit_learn.py`: scikit-learn `LinearRegression` and `SGDRegressor`
  - `ames_linear_regression_project/`: end-to-end project on an Ames housing subset (download, visualize, train)
- **Practice quizzes:** multiple linear regression, gradient descent in practice

### [Week 3: Logistic regression](course1_supervised_learning/w3_logistic_regression/)

- **Optional labs:** classification, sigmoid, decision boundary, logistic loss, cost, gradient descent, scikit-learn, overfitting, regularization
- **My practice:** labs 1 to 9, [`practice.ipynb`](course1_supervised_learning/w3_logistic_regression/labs_practice/practice.ipynb), and [`full_pipeline1.ipynb`](course1_supervised_learning/w3_logistic_regression/labs_practice/full_pipeline1.ipynb), a full scikit-learn pipeline with `GridSearchCV`
- **Practice quizzes:** classification, cost function, gradient descent, overfitting

## Course 2: Advanced Learning Algorithms

Certificate: [`certificate/course2_certificate.pdf`](course2_advanced_learning_algorithms/certificate/course2_certificate.pdf)

### [Week 1: Neural networks](course2_advanced_learning_algorithms/w1_neural_networks/)

- **Optional labs:** neurons and layers, coffee roasting in TensorFlow and in NumPy
- **My practice:** labs 1 to 3
- **Programming assignment:** [`a1_neural_networks_for_handwritten_digit_recognition_binary/`](course2_advanced_learning_algorithms/w1_neural_networks/programming_assignments/a1_neural_networks_for_handwritten_digit_recognition_binary/), with my redo in `assignment1.ipynb`
- **Practice quizzes:** intuition, model, TensorFlow implementation, implementation in Python

### [Week 2: Neural network training](course2_advanced_learning_algorithms/w2_neural_network_training/)

- **Optional labs:** ReLU, softmax, multiclass in TensorFlow (`optional_labs_1_3/`); derivatives and backprop (`optional_labs_4_5/`)
- **My practice:** labs 1 to 3
- **My code:** [`adam_optimizer_intuition.ipynb`](course2_advanced_learning_algorithms/w2_neural_network_training/my_code/adam_optimizer_intuition.ipynb), Adam's moment estimates step by step
- **Programming assignment:** [`a1_multiclass_classification/`](course2_advanced_learning_algorithms/w2_neural_network_training/programming_assignments/a1_multiclass_classification/)
- **Practice quizzes:** training, activation functions, multiclass, additional concepts

### [Week 3: Advice for applying machine learning](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/)

- **Optional labs:** model evaluation and selection, diagnosing bias and variance (my versions: `lab1.ipynb`, `lab2.ipynb`, `utils_practice_version.py`)
- **Programming assignment:** [`a1_advice_for_applying_machine_learning/`](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/programming_assignments/a1_advice_for_applying_machine_learning/), with my redo in `lab3.ipynb`
- **Practice quizzes:** advice, bias and variance, ML development process

### [Week 4: Decision trees](course2_advanced_learning_algorithms/w4_decision_trees/)

- **Optional labs:** decision trees, tree ensembles (my versions: `lab1.ipynb`, `lab2.ipynb`, `utils_practice.py`)
- **Programming assignment:** [`a1_decision_trees/`](course2_advanced_learning_algorithms/w4_decision_trees/programming_assignments/a1_decision_trees/), with my redo in `lab1.ipynb` using my own `utils_practice.py`
- **Practice quizzes:** decision trees, training, tree ensembles

## Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning

In progress.

- [`covariance_ellipses/`](course3_unsupervised_learning_recommenders_rl/covariance_ellipses/): data generation, covariance eigen-ellipses, multi-cluster ellipses. The helpers go in [`mlspec.stats.ellipses`](src/mlspec/stats/ellipses.py).
- [`scratch/`](course3_unsupervised_learning_recommenders_rl/scratch/):
  - `creating_kmeans_bad_local_minima.ipynb`: building data where k-means gets stuck in a bad local minimum
  - `kmeans_plus_plus_scratchwork.ipynb`: k-means++ initialization
  - `feature_scaling_kmeans.ipynb`: how feature scaling changes k-means clusters
  - `kruskal_mst_union_find_and_hdbscan.ipynb`: Kruskal's MST with rank-based union-find, toward HDBSCAN
  - `numpy_views_vs_copies.ipynb`: when NumPy indexing returns a view vs a copy

## Reusable package: `mlspec`

| Module | What it does | Tests |
| ------ | ------------ | ----- |
| [`linear_regression/univariate.py`](src/mlspec/linear_regression/univariate.py) | Cost, gradients, and gradient descent with tolerance-based and patience-based early stopping | [`test_univariate.py`](tests/linear_regression/test_univariate.py), including a numerical gradient check |
| [`viz/matrix_transform_viewer.py`](src/mlspec/viz/matrix_transform_viewer.py) | Interactive view of how a 2×2 matrix transforms the plane (grid, basis vectors, a chosen vector) | [`test_matrix_transform_viewer.py`](tests/viz/test_matrix_transform_viewer.py) |
| [`stats/ellipses.py`](src/mlspec/stats/ellipses.py) | Covariance ellipse helpers (in progress) | to come |

## Books

- *Understanding Deep Learning* (Prince): [`books/understanding_deep_learning/ch01/1_1_background_mathematics.ipynb`](books/understanding_deep_learning/ch01/1_1_background_mathematics.ipynb)

## Scratch

- [`scratch/pandas_tutorial/`](scratch/pandas_tutorial/): pandas practice on the Ames housing subset, plus a vectorization notebook
- [`scratch/w_vs_cost_plot.py`](scratch/w_vs_cost_plot.py): cost as a function of w
