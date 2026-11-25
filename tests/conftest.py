import matplotlib

# Headless backend: plt.show() becomes a no-op, so plotting code runs in tests.
matplotlib.use("Agg")
