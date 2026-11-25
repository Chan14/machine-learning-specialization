## Notes on clustering and the k-means Algorithm

### Part 1 - Clustering

Clustering takes points with no labels and finds interesting patterns in them (puts similar points in groups called clusters).
k-means is one of the algorithms that does clustering. Following sections apply to k-means.

### k-means - 1. The Algorithm

1. Each cluster is represented by a centroid, which is a location in the data's space.
2. k-means needs $K$ given in advance.

3. Type of algorithm: k-means is a **hard EM** (hard expectation–maximization) algorithm, which is a form of **alternating minimization**. It keeps two sets of unknowns: which cluster each point belongs to, and where each centroid is. It never improves both at once. Instead it alternates two steps: hold the centroids fixed and give every point its best cluster, then hold those assignments fixed and move every centroid to its best location. Each step jumps straight to the best answer for the variables it changes. It's called _hard_ EM because each point belongs to exactly one cluster. Soft EM, as in Gaussian mixture models, gives each point a probability of belonging to each cluster. There are no derivatives, gradients, or learning rate: the best assignment is just the nearest centroid, and the best centroid is just the mean of its points.

4. Assign step - Assign points to clusters keeping the centroids fixed - each point gets assigned to its nearest centroid by squared distance, keeping the centroids fixed.
5. Update step - Move centroids to the mean of the points assigned to it keeping the centroid assignments to the points fixed.
6. Repeat until no assignment changes. If a cluster ends up with no points, either drop it or restart its centroid at a random point.

### k-means - 2. The Cost Function and why the loop works

**Symbols**

- $m$: number of points. $K$: number of clusters.
- $x^{(i)}$: point $i$.
- $c^{(i)}$: the cluster point $i$ is assigned to, a number from $1$ to $K$.
- $\mu_k$: the centroid of cluster $k$, a location in the same space as the points.
- $\mu_{c^{(i)}}$: the centroid of point $i$'s own cluster.
- $C_k$: the set of points in cluster $k$. $|C_k|$: how many points that is.
- $\bar x_k$: the mean of cluster $k$'s points.

**1. The Cost Function**
$$J = \frac{1}{m}\sum_{i=1}^{m} \bigl\| x^{(i)} - \mu_{c^{(i)}} \bigr\|^2$$
$J$ is the average squared distance from each point to its own centroid. It's also called the distortion. `scikit-learn` calls the sum without the $\frac1m$ `inertia_`, and statistics calls it the within-cluster sum of squares.

**2. k-means minimizes $J$ by block coordinate descent.** $J$ depends on two blocks of unknowns: the assignments $c^{(1)}, \dots, c^{(m)}$ and the centroids $\mu_1, \dots, \mu_K$. Each step holds one block fixed and moves the other to the value that makes $J$ smallest. This general optimization strategy is called (block) coordinate descent, and this assign-then-update loop is known as Lloyd's algorithm, the standard k-means algorithm. Unlike gradient descent, it needs no gradient and no learning rate. That matters because the assignments are discrete labels from $1$ to $K$ with nothing in between, so $J$ has no derivative with respect to them.

**3. The assign step never increases $J$.** With the centroids fixed, a point switches its assignment only if the new centroid is strictly closer than its current one. Each switch strictly lowers $J$, and points that don't switch leave $J$ unchanged, so $J$ can only go down or stay the same.

**4. The mean is the best spot for a centroid.** For any group of $n$ points $y_1, \dots, y_n$ with mean $\bar y$, and any location $\mu$:

$$\sum_{j=1}^{n} \|y_j - \mu\|^2 = \sum_{j=1}^{n} \|y_j - \bar y\|^2 + n\,\|\bar y - \mu\|^2$$

The total squared distance from the points to $\mu$ equals their total squared distance to the mean, plus a penalty of $n$ times the squared distance between $\mu$ and the mean. The penalty is zero only at the mean, so the mean gives the smallest total, which is why the update step moves each centroid to its cluster's mean.

The identity holds because the deviations of any set of points from their own mean sum up to zero. That is true for every dataset, whatever the data.

The mean is the best spot only when distance is measured as squared distance. For plain (unsquared) distance, the best spot is the geometric median. For absolute differences summed per coordinate, it's the per-coordinate median, which gives k-medians.

**5. The update step never increases $J$.** With the assignments fixed, $J$ splits into one piece per cluster. Moving each centroid to its cluster's mean gives each piece its smallest value.

**6. The loop always stops.** A sequence that strictly drops at every step, and can only take values from a finite set, can take only finitely many steps.

k-means fits this. Each of the $m$ points can be assigned to any of the $K$ clusters, so there are $K^m$ possible assignment lists. That's a huge number, but it's finite. Once you know the assignment list, you know the centroids after the update step, and therefore the cost, so each assignment list gives exactly one cost. The loop only changes the assignments when doing so strictly lowers the cost. When no change would lower it, nothing changes, and the loop has converged.

So the cost strictly drops at every round that changes the assignments, and it can only take finitely many values. The algorithm can never return to an assignment list it has already used, because that would bring back a higher cost. With only $K^m$ lists, the rounds that change the assignments must run out, and the loop stops.

**7. A point can be equally close to two or more centroids, i.e., Ties can happen.** With the rule "switch only if the new centroid is strictly closer," the assign step itself strictly lowers $J$ whenever anything changes. With NumPy's `argmin`, which picks the lowest cluster index on a tie, a tied point may switch clusters while the assign step leaves $J$ unchanged, but the update step that follows then strictly lowers it. Either way the loop eventually stops.

The update step strictly lowers $J$ because removing the tied point from its old cluster moves that cluster's mean, so the old centroid is no longer at its cluster's mean, and moving it there strictly lowers $J$ (point 4). This fails only if the tied point sits exactly on two centroids that share the same spot. Even then, `argmin` picks the same cluster every round, so the loop still stops. Only a tie rule that can choose differently in different rounds, like random tie-breaking, could loop forever.

**8. Iterations can be capped to not waste compute on diminishing returns.** The guaranteed bound of $K^m$ rounds is astronomically large. A few rare point sets really do need a huge number of rounds. Most of the drop in $J$ happens in the first few rounds and the rest is a long tail of tiny improvements. And rounding errors can make a nearly-tied point flip back and forth forever. In practice, `scikit-learn` stops when the centroids barely move (`tol`) or after `max_iter=300` rounds.

**9. The local minima.** Local minima are the locations of sub-optimal cost in the parameter space that the K-means cannot escape from.
The loop stops when neither the assign step nor the update step changes anything. That can happen at a local minimum: a clustering that neither step can improve, but whose $J$ is much higher than the best possible.

The typical local minimum pattern: two centroids share one true group, while one centroid sits in the empty space between two other groups. The steps can't escape, because the assign step judges each switch only against where the centroids are right now. Escaping would need several points and a centroid to move together, and every single switch, judged at the current positions, raises $J$. Finding the clustering with the truly best $J$ is NP-hard. The starting centroids decide which local minimum you reach.

**10. $J$ is the size-weighted average of the within-cluster variances.** A cluster's total variance $\text{Var}_k$ is the sum of its variances across all the feature dimensions (for example, for 2D points, it is the sum of variances in the $x$ and the $y$ directions). Right after an update step:
$$J = \sum_{k=1}^{K} \frac{|C_k|}{m}\,\text{Var}_k$$
Each cluster's variance is weighted by the fraction of points it holds, and the weights add up to 1. So k-means makes clusters as tight as possible, with bigger clusters counting more toward $J$.

The same total variance is the trace of the cluster's covariance matrix $\Sigma_k$. The diagonal of $\Sigma_k$ holds each feature's variance, and the trace adds those up, so $\text{Var}_k = \mathrm{tr}(\Sigma_k)$. k-means only ever sees this trace. The off-diagonal entries of $\Sigma_k$, which describe how a cluster is tilted or stretched, do not affect $J$.

Implementation detail: use the divide-by-$n$ versions when calculating variances. `np.var` divides by $n$ by default, while `torch.var` needs `correction=0` (default is correction=1). For covariance matrices, both `np.cov` and `torch.cov` divide by $n-1$ by default, so use `np.cov(..., bias=True)` or `torch.cov(..., correction=0)`.

**11. What k-means assumes.** The algorithm splits two clusters exactly at the midpoint between their centroids. The boundary between two clusters is a flat surface that cuts the line joining the two centroids at its midpoint, at a right angle. It's called a hyperplane, and in 2D it's the perpendicular bisector of the line joining the two centroids. Together, these boundaries cut the space into one region per centroid, called a Voronoi diagram. The boundary depends only on where the centroids are, never on how spread out or stretched the clusters are. So k-means can split a stretched cluster midway and give its far end to a neighboring centroid, because those far points are closer to the neighbor's centroid than to their own cluster's centroid. In the same way, when a wide cluster sits next to a tight one, the boundary still sits halfway between the centroids, so the wide cluster's outer points that reach past the midpoint are given to the tight cluster. The right boundary would sit closer to the tight cluster.

So k-means works best when the clusters are roughly round, well-separated, and have equal size and spread. With stretched clusters, or clusters with very different spreads, the lowest-$J$ grouping itself is usually wrong, so more random starts don't help. Practical fixes: scale features to similar ranges. For embeddings, normalize each vector to length 1. This removes stretching along the line from the origin, which comes from length differences that don't carry meaning, and makes k-means's distance rank points by cosine similarity. Or use methods that model each cluster's shape, such as Gaussian mixtures, which give each cluster its own spread and tilt, described by a covariance matrix.
