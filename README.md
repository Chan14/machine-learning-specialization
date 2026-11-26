# machine-learning-specialization

My work through the [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) (DeepLearning.AI and Stanford Online, taught by Andrew Ng): the course labs, my own re-implementations and experiments, and the reusable code that came out of them.

**Jump to:** [Course 1](#course-1--supervised-machine-learning-regression-and-classification) · [Course 2](#course-2--advanced-learning-algorithms) · [Course 3](#course-3--unsupervised-learning-recommenders-reinforcement-learning) · [Beyond the courses](#beyond-the-courses) · [Working in this repo](#working-in-this-repo)

In each week, the course's own material comes first. **My practice** (re-implementations) and **My code** (my own scripts and projects) follow it.

## Course 1 : [Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning)

- [Week 1 : Introduction to machine learning](course1_supervised_learning/w1_univariate_linear_regression)
  - [Practice quiz: Supervised vs unsupervised learning](course1_supervised_learning/w1_univariate_linear_regression/practice_quizzes/p1_supervised_vs_unsupervised_learning)
  - [Practice quiz: Regression](course1_supervised_learning/w1_univariate_linear_regression/practice_quizzes/p2_regression)
  - [Practice quiz: Train the model with gradient descent](course1_supervised_learning/w1_univariate_linear_regression/practice_quizzes/p3_train_the_model_with_gradient_descent)
  - [Optional Labs](course1_supervised_learning/w1_univariate_linear_regression/optional_labs)
    - [Python and Jupyter](course1_supervised_learning/w1_univariate_linear_regression/optional_labs/C1_W1_Lab01_Python_Jupyter_Soln.ipynb)
    - [Model Representation](course1_supervised_learning/w1_univariate_linear_regression/optional_labs/C1_W1_Lab02_Model_Representation_Soln.ipynb)
    - [Cost Function](course1_supervised_learning/w1_univariate_linear_regression/optional_labs/C1_W1_Lab03_Cost_function_Soln.ipynb)
    - [Gradient Descent](course1_supervised_learning/w1_univariate_linear_regression/optional_labs/C1_W1_Lab04_Gradient_Descent_Soln.ipynb)
  - [My practice](course1_supervised_learning/w1_univariate_linear_regression/labs_practice): [Lab 2](course1_supervised_learning/w1_univariate_linear_regression/labs_practice/lab2_practice.ipynb) · [Lab 3](course1_supervised_learning/w1_univariate_linear_regression/labs_practice/lab3_practice.ipynb) · [Lab 4](course1_supervised_learning/w1_univariate_linear_regression/labs_practice/lab4_practice.ipynb)
  - [My code](course1_supervised_learning/w1_univariate_linear_regression/my_code)
    - [Learning rate experiments](course1_supervised_learning/w1_univariate_linear_regression/my_code/learning_rate_experiments.py): learning rates, early stopping, and divergence on synthetic data (model: [mlspec.linear_regression.univariate](src/mlspec/linear_regression/univariate.py))
    - [Plotters](course1_supervised_learning/w1_univariate_linear_regression/my_code/plotters.py): loss curves and model-fit plots

- [Week 2 : Regression with multiple input variables](course1_supervised_learning/w2_multivariate_linear_regression)
  - [Practice quiz: Multiple linear regression](course1_supervised_learning/w2_multivariate_linear_regression/practice_quizzes/p1_multiple_linear_regression)
  - [Practice quiz: Gradient descent in practice](course1_supervised_learning/w2_multivariate_linear_regression/practice_quizzes/p2_gradient_descent_in_practice)
  - [Optional Labs](course1_supervised_learning/w2_multivariate_linear_regression/optional_labs)
    - [NumPy Vectorization](course1_supervised_learning/w2_multivariate_linear_regression/optional_labs/C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb)
    - [Multiple Variable Regression](course1_supervised_learning/w2_multivariate_linear_regression/optional_labs/C1_W2_Lab02_Multiple_Variable_Soln.ipynb)
    - [Feature Scaling and Learning Rate](course1_supervised_learning/w2_multivariate_linear_regression/optional_labs/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb)
    - [Feature Engineering and Polynomial Regression](course1_supervised_learning/w2_multivariate_linear_regression/optional_labs/C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb)
    - [Scikit-learn Gradient Descent](course1_supervised_learning/w2_multivariate_linear_regression/optional_labs/C1_W2_Lab05_Sklearn_GD_Soln.ipynb)
    - [Scikit-learn Normal Equation](course1_supervised_learning/w2_multivariate_linear_regression/optional_labs/C1_W2_Lab06_Sklearn_Normal_Soln.ipynb)
  - [My practice](course1_supervised_learning/w2_multivariate_linear_regression/labs_practice): [Lab 1](course1_supervised_learning/w2_multivariate_linear_regression/labs_practice/lab1_practice.ipynb) · [Lab 2](course1_supervised_learning/w2_multivariate_linear_regression/labs_practice/lab2_practice.ipynb) · [Lab 3](course1_supervised_learning/w2_multivariate_linear_regression/labs_practice/lab3_practice.ipynb) · [Lab 4](course1_supervised_learning/w2_multivariate_linear_regression/labs_practice/lab4_practice.ipynb) · [Lab 5](course1_supervised_learning/w2_multivariate_linear_regression/labs_practice/lab5_practice.ipynb) · [Linear regression mini-project](course1_supervised_learning/w2_multivariate_linear_regression/labs_practice/lr_with_alex.ipynb)
  - [My code](course1_supervised_learning/w2_multivariate_linear_regression/my_code)
    - [Simple multivariate regression](course1_supervised_learning/w2_multivariate_linear_regression/my_code/simple_mv_lin_regression.py): loop vs vectorized prediction, cost, and gradient
    - [Feature scaling and learning curves](course1_supervised_learning/w2_multivariate_linear_regression/my_code/feature_scaling_learning_curve.py)
    - [Feature engineering and polynomial regression](course1_supervised_learning/w2_multivariate_linear_regression/my_code/feature_eng_and_poly_reg.py)
    - [Scikit-learn LinearRegression](course1_supervised_learning/w2_multivariate_linear_regression/my_code/lin_regression_lr_scikit_learn.py)
    - [Scikit-learn SGDRegressor](course1_supervised_learning/w2_multivariate_linear_regression/my_code/lin_regression_sgrd_scikit_learn.py)
    - [Ames housing project](course1_supervised_learning/w2_multivariate_linear_regression/my_code/ames_linear_regression_project): download, visualize, and train on an Ames housing subset

- [Week 3 : Classification](course1_supervised_learning/w3_logistic_regression)
  - [Practice quiz: Classification with logistic regression](course1_supervised_learning/w3_logistic_regression/practice_quizzes/p1_classification_with_logistic_regression)
  - [Practice quiz: Cost function for logistic regression](course1_supervised_learning/w3_logistic_regression/practice_quizzes/p2_cost_function_for_logistic_regression)
  - [Practice quiz: Gradient descent for logistic regression](course1_supervised_learning/w3_logistic_regression/practice_quizzes/p3_gradient_descent_for_logistic_regression)
  - [Practice quiz: The problem of overfitting](course1_supervised_learning/w3_logistic_regression/practice_quizzes/p4_the_problem_of_overfitting)
  - [Optional Labs](course1_supervised_learning/w3_logistic_regression/optional_labs)
    - [Classification](course1_supervised_learning/w3_logistic_regression/optional_labs/C1_W3_Lab01_Classification_Soln.ipynb)
    - [Sigmoid Function](course1_supervised_learning/w3_logistic_regression/optional_labs/C1_W3_Lab02_Sigmoid_function_Soln.ipynb)
    - [Decision Boundary](course1_supervised_learning/w3_logistic_regression/optional_labs/C1_W3_Lab03_Decision_Boundary_Soln.ipynb)
    - [Logistic Loss](course1_supervised_learning/w3_logistic_regression/optional_labs/C1_W3_Lab04_LogisticLoss_Soln.ipynb)
    - [Cost Function](course1_supervised_learning/w3_logistic_regression/optional_labs/C1_W3_Lab05_Cost_Function_Soln.ipynb)
    - [Gradient Descent](course1_supervised_learning/w3_logistic_regression/optional_labs/C1_W3_Lab06_Gradient_Descent_Soln.ipynb)
    - [Scikit-learn Logistic Regression](course1_supervised_learning/w3_logistic_regression/optional_labs/C1_W3_Lab07_Scikit_Learn_Soln.ipynb)
    - [Overfitting](course1_supervised_learning/w3_logistic_regression/optional_labs/C1_W3_Lab08_Overfitting_Soln.ipynb)
    - [Regularization](course1_supervised_learning/w3_logistic_regression/optional_labs/C1_W3_Lab09_Regularization_Soln.ipynb)
  - [My practice](course1_supervised_learning/w3_logistic_regression/labs_practice): [Lab 1](course1_supervised_learning/w3_logistic_regression/labs_practice/lab1_practice.ipynb) · [Lab 2](course1_supervised_learning/w3_logistic_regression/labs_practice/lab2_practice.ipynb) · [Lab 3](course1_supervised_learning/w3_logistic_regression/labs_practice/lab3_practice.ipynb) · [Lab 4](course1_supervised_learning/w3_logistic_regression/labs_practice/lab4_practice.ipynb) · [Lab 5](course1_supervised_learning/w3_logistic_regression/labs_practice/lab5_practice.ipynb) · [Lab 6](course1_supervised_learning/w3_logistic_regression/labs_practice/lab6_practice.ipynb) · [Lab 7](course1_supervised_learning/w3_logistic_regression/labs_practice/lab7_practice.ipynb) · [Lab 8](course1_supervised_learning/w3_logistic_regression/labs_practice/lab8_practice.ipynb) · [Lab 9](course1_supervised_learning/w3_logistic_regression/labs_practice/lab9_practice.ipynb) · [Practice](course1_supervised_learning/w3_logistic_regression/labs_practice/practice.ipynb) · [Full pipeline with GridSearchCV](course1_supervised_learning/w3_logistic_regression/labs_practice/full_pipeline1.ipynb)

**[Certificate of Completion](course1_supervised_learning/certificate/course1_certificate.pdf)**

## Course 2 : [Advanced Learning Algorithms](https://www.coursera.org/learn/advanced-learning-algorithms)

- [Week 1 : Neural networks](course2_advanced_learning_algorithms/w1_neural_networks)
  - [Practice quiz: Neural network intuition](course2_advanced_learning_algorithms/w1_neural_networks/practice_quizzes/p1_neural_network_intuition)
  - [Practice quiz: Neural network model](course2_advanced_learning_algorithms/w1_neural_networks/practice_quizzes/p2_neural_network_model)
  - [Practice quiz: TensorFlow implementation](course2_advanced_learning_algorithms/w1_neural_networks/practice_quizzes/p3_tensorflow_implementation)
  - [Practice quiz: Neural network implementation in Python](course2_advanced_learning_algorithms/w1_neural_networks/practice_quizzes/p4_neural_network_implementation_in_python)
  - [Optional Labs](course2_advanced_learning_algorithms/w1_neural_networks/optional_labs)
    - [Neurons and Layers](course2_advanced_learning_algorithms/w1_neural_networks/optional_labs/C2_W1_Lab01_Neurons_and_Layers.ipynb)
    - [Coffee Roasting in TensorFlow](course2_advanced_learning_algorithms/w1_neural_networks/optional_labs/C2_W1_Lab02_CoffeeRoasting_TF.ipynb)
    - [Coffee Roasting in NumPy](course2_advanced_learning_algorithms/w1_neural_networks/optional_labs/C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb)
  - [Programming Assignment](course2_advanced_learning_algorithms/w1_neural_networks/programming_assignments/a1_neural_networks_for_handwritten_digit_recognition_binary)
    - [Neural Networks for Handwritten Digit Recognition, Binary](course2_advanced_learning_algorithms/w1_neural_networks/programming_assignments/a1_neural_networks_for_handwritten_digit_recognition_binary/C2_W1_Assignment.ipynb)
    - My redo: [assignment1.ipynb](course2_advanced_learning_algorithms/w1_neural_networks/programming_assignments/a1_neural_networks_for_handwritten_digit_recognition_binary/assignment1.ipynb)
  - [My practice](course2_advanced_learning_algorithms/w1_neural_networks/labs_practice): [Lab 1](course2_advanced_learning_algorithms/w1_neural_networks/labs_practice/lab1.ipynb) · [Lab 2](course2_advanced_learning_algorithms/w1_neural_networks/labs_practice/lab2.ipynb) · [Lab 3](course2_advanced_learning_algorithms/w1_neural_networks/labs_practice/lab3.ipynb)

- [Week 2 : Neural network training](course2_advanced_learning_algorithms/w2_neural_network_training)
  - [Practice quiz: Neural network training](course2_advanced_learning_algorithms/w2_neural_network_training/practice_quizzes/p1_neural_network_training)
  - [Practice quiz: Activation functions](course2_advanced_learning_algorithms/w2_neural_network_training/practice_quizzes/p2_activation_functions)
  - [Practice quiz: Multiclass classification](course2_advanced_learning_algorithms/w2_neural_network_training/practice_quizzes/p3_multiclass_classification)
  - [Practice quiz: Additional neural network concepts](course2_advanced_learning_algorithms/w2_neural_network_training/practice_quizzes/p4_additional_neural_network_concepts)
  - [Optional Labs](course2_advanced_learning_algorithms/w2_neural_network_training/optional_labs_1_3)
    - [ReLU](course2_advanced_learning_algorithms/w2_neural_network_training/optional_labs_1_3/C2_W2_Relu.ipynb)
    - [Softmax](course2_advanced_learning_algorithms/w2_neural_network_training/optional_labs_1_3/C2_W2_SoftMax.ipynb)
    - [Multiclass Classification](course2_advanced_learning_algorithms/w2_neural_network_training/optional_labs_1_3/C2_W2_Multiclass_TF.ipynb)
  - [Optional Labs](course2_advanced_learning_algorithms/w2_neural_network_training/optional_labs_4_5)
    - [Derivatives](course2_advanced_learning_algorithms/w2_neural_network_training/optional_labs_4_5/C2_W2_Derivatives.ipynb)
    - [Backpropagation](course2_advanced_learning_algorithms/w2_neural_network_training/optional_labs_4_5/C2_W2_Backprop.ipynb)
  - [Programming Assignment](course2_advanced_learning_algorithms/w2_neural_network_training/programming_assignments/a1_multiclass_classification)
    - [Neural Networks for Handwritten Digit Recognition, Multiclass](course2_advanced_learning_algorithms/w2_neural_network_training/programming_assignments/a1_multiclass_classification/C2_W2_Assignment.ipynb)
  - [My practice](course2_advanced_learning_algorithms/w2_neural_network_training/labs_practice): [Lab 1](course2_advanced_learning_algorithms/w2_neural_network_training/labs_practice/lab1.ipynb) · [Lab 2](course2_advanced_learning_algorithms/w2_neural_network_training/labs_practice/lab2.ipynb) · [Lab 3](course2_advanced_learning_algorithms/w2_neural_network_training/labs_practice/lab3.ipynb)
  - [My code](course2_advanced_learning_algorithms/w2_neural_network_training/my_code)
    - [Adam optimizer intuition](course2_advanced_learning_algorithms/w2_neural_network_training/my_code/adam_optimizer_intuition.ipynb): Adam's moment estimates step by step

- [Week 3 : Advice for applying machine learning](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning)
  - [Practice quiz: Advice for applying machine learning](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/practice_quizzes/p1_advice_for_applying_machine_learning)
  - [Practice quiz: Bias and variance](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/practice_quizzes/p2_bias_and_variance)
  - [Practice quiz: Machine learning development process](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/practice_quizzes/p3_machine_learning_development_process)
  - [Optional Labs](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/optional_labs)
    - [Model Evaluation and Selection](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/optional_labs/C2W3_Lab_01_Model_Evaluation_and_Selection.ipynb)
    - [Diagnosing Bias and Variance](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/optional_labs/C2W3_Lab_02_Diagnosing_Bias_and_Variance.ipynb)
  - [Programming Assignment](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/programming_assignments/a1_advice_for_applying_machine_learning)
    - [Advice for Applied Machine Learning](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/programming_assignments/a1_advice_for_applying_machine_learning/C2_W3_Assignment.ipynb)
    - My redo: [lab3.ipynb](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/programming_assignments/a1_advice_for_applying_machine_learning/lab3.ipynb) · [assignment_utils_practice.py](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/programming_assignments/a1_advice_for_applying_machine_learning/assignment_utils_practice.py)
  - [My practice](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/optional_labs): [Lab 1](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/optional_labs/lab1.ipynb) · [Lab 2](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/optional_labs/lab2.ipynb) · [utils_practice_version.py](course2_advanced_learning_algorithms/w3_advice_for_applying_machine_learning/optional_labs/utils_practice_version.py)

- [Week 4 : Decision trees](course2_advanced_learning_algorithms/w4_decision_trees)
  - [Practice quiz: Decision trees](course2_advanced_learning_algorithms/w4_decision_trees/practice_quizzes/p1_decision_trees)
  - [Practice quiz: Decision tree training](course2_advanced_learning_algorithms/w4_decision_trees/practice_quizzes/p2_decision_tree_training)
  - [Practice quiz: Tree ensembles](course2_advanced_learning_algorithms/w4_decision_trees/practice_quizzes/p3_tree_ensembles)
  - [Optional Labs](course2_advanced_learning_algorithms/w4_decision_trees/optional_labs)
    - [Decision Trees](course2_advanced_learning_algorithms/w4_decision_trees/optional_labs/C2_W4_Lab_01_Decision_Trees.ipynb)
    - [Tree Ensembles](course2_advanced_learning_algorithms/w4_decision_trees/optional_labs/C2_W4_Lab_02_Tree_Ensemble.ipynb)
  - [Programming Assignment](course2_advanced_learning_algorithms/w4_decision_trees/programming_assignments/a1_decision_trees)
    - [Decision Trees](course2_advanced_learning_algorithms/w4_decision_trees/programming_assignments/a1_decision_trees/C2_W4_Decision_Tree_with_Markdown.ipynb)
    - My redo: [lab1.ipynb](course2_advanced_learning_algorithms/w4_decision_trees/programming_assignments/a1_decision_trees/lab1.ipynb) · [utils_practice.py](course2_advanced_learning_algorithms/w4_decision_trees/programming_assignments/a1_decision_trees/utils_practice.py)
  - [My practice](course2_advanced_learning_algorithms/w4_decision_trees/optional_labs): [Lab 1](course2_advanced_learning_algorithms/w4_decision_trees/optional_labs/lab1.ipynb) · [Lab 2](course2_advanced_learning_algorithms/w4_decision_trees/optional_labs/lab2.ipynb) · [utils_practice.py](course2_advanced_learning_algorithms/w4_decision_trees/optional_labs/utils_practice.py)

**[Certificate of Completion](course2_advanced_learning_algorithms/certificate/course2_certificate.pdf)**

## Course 3 : [Unsupervised Learning, Recommenders, Reinforcement Learning](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning)

In progress.

- [Week 1 : Unsupervised learning](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning)
  - [Clustering](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering)
    - [Practice quiz: Clustering](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/practice_quizzes/p1_clustering)
    - [My notes](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/notes)
      - [k-means rolling summary, Parts 1–2: the algorithm and the cost function](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/notes/kmeans_rolling_summary_1.md)
      - [k-means rolling summary, Parts 3–4: starting centroids and choosing K](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/notes/kmeans_rolling_summary_2.md)
    - [My code](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/my_code)
      - [Covariance ellipses](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/my_code/covariance_ellipses): [Data generation](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/my_code/covariance_ellipses/01_data_generation.ipynb) · [Covariance eigen-ellipse](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/my_code/covariance_ellipses/02_cov_eigen_ellipse.ipynb) · [Multi-cluster ellipses](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/my_code/covariance_ellipses/03_multi_cluster.ipynb)
      - [Scratch work](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/my_code/scratch)
        - [Creating bad k-means local minima](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/my_code/scratch/creating_kmeans_bad_local_minima.ipynb)
        - [k-means++ initialization](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/my_code/scratch/kmeans_plus_plus_scratchwork.ipynb)
        - [Feature scaling and k-means](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/my_code/scratch/feature_scaling_kmeans.ipynb)
        - [Kruskal's MST, union-find, and HDBSCAN](course3_unsupervised_learning_recommenders_rl/w1_unsupervised_learning/clustering/my_code/scratch/kruskal_mst_union_find_and_hdbscan.ipynb)
  - Anomaly detection (coming next)
- [Course 3 scratch](course3_unsupervised_learning_recommenders_rl/scratch)
  - [NumPy views vs copies](course3_unsupervised_learning_recommenders_rl/scratch/numpy_views_vs_copies.ipynb)

## Beyond the courses

### Reusable package: `mlspec`

| Module                                                                        | What it does                                                                                 | Tests                                                                                                  |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [linear_regression/univariate.py](src/mlspec/linear_regression/univariate.py) | Cost, gradients, and gradient descent with tolerance-based and patience-based early stopping | [test_univariate.py](tests/linear_regression/test_univariate.py), including a numerical gradient check |
| [viz/matrix_transform_viewer.py](src/mlspec/viz/matrix_transform_viewer.py)   | Interactive view of how a 2×2 matrix transforms the plane                                    | [test_matrix_transform_viewer.py](tests/viz/test_matrix_transform_viewer.py)                           |
| [stats/ellipses.py](src/mlspec/stats/ellipses.py)                             | Covariance ellipse helpers (in progress)                                                     | to come                                                                                                |

### Books

- _Understanding Deep Learning_ (Prince)
  - [Chapter 1: Background mathematics](books/understanding_deep_learning/ch01/1_1_background_mathematics.ipynb)

### Scratch

- [Pandas tutorial](scratch/pandas_tutorial): pandas practice on the Ames housing subset, plus a [vectorization notebook](scratch/pandas_tutorial/vectorization/vectorize.ipynb)
- [Cost vs w plot](scratch/w_vs_cost_plot.py)

## Working in this repo

### Repository layout

```
machine-learning-specialization/
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

| Folder                               | What's in it                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------ |
| `optional_labs/`                     | Course-provided labs (solutions), with their `lab_utils*.py`, images, and data |
| `labs_practice/`                     | My own re-implementations of those labs                                        |
| `programming_assignments/aN_<name>/` | Graded assignments, each with its own utilities and tests                      |
| `practice_quizzes/pN_<topic>/`       | Screenshots of practice quizzes                                                |
| `my_code/`                           | My own scripts and mini-projects for that week                                 |
| `notes/`                             | My rolling summaries of the topic, in Markdown with math                       |
| `archive*/`                          | Older versions kept for reference                                              |

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

### Setup

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

### Everyday commands

```bash
pyenv activate ml-spec-env                     # once per terminal
pytest                                         # run all tests
python course1_supervised_learning/w1_univariate_linear_regression/my_code/learning_rate_experiments.py
ruff check src tests                           # lint my package
```

### Where new work goes

| Content type...                           | Where it goes                         |
| ----------------------------------------- | ------------------------------------- |
| Course lab files (downloaded)             | `courseN_.../wN_.../optional_labs/`   |
| My re-implementation of a lab             | `courseN_.../wN_.../labs_practice/`   |
| My own script or mini-project for a week  | `courseN_.../wN_.../my_code/`         |
| A function I'll reuse across notebooks    | `src/mlspec/...` + `tests/...`        |
| Rough work for a specific course          | `courseN_.../scratch/`                |
| Rough work not tied to a course           | `scratch/`                            |
| Anything that must never reach GitHub     | any `untracked/` folder               |
| My notes or rolling summaries for a topic | `courseN_.../wN_.../<section>/notes/` |
