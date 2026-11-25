import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlspec.viz.matrix_transform_viewer import visualize_transformation


@pytest.mark.parametrize(
    "M",
    [
        np.array([[2, 1], [0, 1]]),  # shear + stretch
        np.array([[0, -1], [1, 0]]),  # 90-degree rotation
        np.array([[1, 1], [2, 2]]),  # singular: collapses the plane to a line
    ],
)
def test_draws_without_error(M):
    visualize_transformation(M, show_old_grid=True)
    plt.close("all")
