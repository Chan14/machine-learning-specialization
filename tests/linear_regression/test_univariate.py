import numpy as np
import pytest

from mlspec.linear_regression.univariate import (
    compute_gradients,
    compute_loss,
    gradient_descent_patience_based,
    gradient_descent_with_loss_gradient_tol,
)


@pytest.fixture
def line_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-10, 10, size=200)
    Y = 2.5 * X - 1.0 + rng.normal(0, 0.5, size=200)
    return X, Y


def test_loss_is_half_mse():
    X = np.array([0.0, 1.0, 2.0])
    Y = np.array([1.0, 1.0, 1.0])
    # w=0, b=0 -> errors are all -1 -> MSE = 1 -> J = 1/2
    assert compute_loss(X, Y, 0.0, 0.0) == pytest.approx(0.5)


@pytest.mark.parametrize("w, b", [(0.0, 0.0), (1.3, -0.7), (-2.0, 4.0)])
def test_gradients_match_numerical_derivative_of_loss(line_data, w, b):
    # Invariant: compute_gradients is the derivative of compute_loss.
    # Central differences: (J(w+eps) - J(w-eps)) / (2*eps).
    X, Y = line_data
    eps = 1e-6
    dw_num = (compute_loss(X, Y, w + eps, b) - compute_loss(X, Y, w - eps, b)) / (2 * eps)
    db_num = (compute_loss(X, Y, w, b + eps) - compute_loss(X, Y, w, b - eps)) / (2 * eps)
    dw, db = compute_gradients(X, Y, w, b)
    assert dw == pytest.approx(dw_num, rel=1e-5)
    assert db == pytest.approx(db_num, rel=1e-5)


def test_gradient_descent_recovers_parameters(line_data):
    X, Y = line_data
    losses, grad_norms, ws, bs = gradient_descent_with_loss_gradient_tol(
        X, Y, learning_rate=0.01, max_iters=5000, loss_tol=0, grad_tol=1e-8,
        verbose=False,
    )
    w_ls, b_ls = np.polyfit(X, Y, 1)  # closed-form least squares
    assert ws[-1] == pytest.approx(w_ls, abs=1e-4)
    assert bs[-1] == pytest.approx(b_ls, abs=1e-4)
    assert len(losses) == len(grad_norms) == len(ws) == len(bs)


def test_loss_never_increases_for_small_learning_rate(line_data):
    X, Y = line_data
    losses, *_ = gradient_descent_with_loss_gradient_tol(
        X, Y, learning_rate=0.005, max_iters=300, verbose=False
    )
    assert np.all(np.diff(losses) <= 1e-12)


def test_patience_stops_early_when_validation_stalls(line_data):
    X, Y = line_data
    w, b, losses = gradient_descent_patience_based(
        X, Y, learning_rate=0.01, max_iters=10_000, val_X=X, val_Y=Y,
        val_patience=5, min_delta=1e-3, verbose=False,
    )
    assert len(losses) < 10_000
    assert w == pytest.approx(2.5, abs=0.1)
