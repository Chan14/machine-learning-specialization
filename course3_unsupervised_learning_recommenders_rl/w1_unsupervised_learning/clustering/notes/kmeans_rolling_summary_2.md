## Notes on clustering and the k-means Algorithm

### k-means - 3. Choosing the starting centroids (or the seeds)

**Symbols**

- **Seed**: a starting centroid, chosen before the k-means iterations begin.
- **Blob**: one true, well-separated group of points in the data.
- $D_i^2$: point $i$'s squared distance to its nearest seed chosen so far.
- $p$: the probability that one k-means run ends with the right clusters.
- $n_{\text{runs}}$: the number of independent k-means runs.
- $w_c$: the average $D_i^2$ of a point in a covered blob. $w_u$: the average $D_i^2$ of a point in an uncovered blob.

**1. The simplest way of picking the starting centroids - random initialization.**
The course picks $K$ random data points as seeds. The seeds are $K$ distinct data points, not random locations in the space. A seed that is a data point always has at least one point assigned to it, itself at distance 0, so no cluster is empty after the first assign step.

Gotchas: sample with `replace=False`, otherwise the same point can be picked twice. Even then, duplicate rows in the data can put two seeds on the same spot. And if you take the seeds with a slice such as `X[:K]` after shuffling, you get a view of `X`, so updating the centroids in place would change the data. Integer-array indexing like `X[idx]` already returns a copy.

**2. Random seeds often put two seeds in the same blob and leave another blob with none.** With $K$ equal-sized blobs, the chance that every blob gets exactly one seed is
$$\frac{K}{K}\cdot\frac{K-1}{K}\cdots\frac{1}{K} = \frac{K!}{K^K}$$
Each factor is the chance that the next seed lands in an uncovered blob, given that the earlier seeds all landed in different blobs: the first seed can land anywhere, the second must avoid one blob, and so on. That's about 9% for $K = 4$ and about one in a million for $K = 16$. A run that starts with two seeds in one blob sometimes still ends with the right clusters, because the iterations can move one of those seeds over to the blob with no seed. So the real success rate is higher than $K!/K^K$, but it still falls quickly as $K$ grows: measured at 62%, 12% and 1.6% for $K = 4, 9, 16$.

**3. The course's fix is to restart many times and keep the run with the lowest $J$.** The course runs k-means 50 to 1000 times from different random seeds. If one run succeeds with probability $p$, then $n$ runs all fail with probability $(1-p)^n$. For a 99% chance that at least one run succeeds, that failure chance must be at most $0.01$, which gives
$$n_{\text{runs}} \ge \frac{\ln 0.01}{\ln(1-p)}$$
The inequality flips from $\le$ to $\ge$ because $\ln(1-p)$ is negative, and dividing by a negative number flips it. Measured: 5 runs for $K = 4$, 35 for $K = 9$, 286 for $K = 16$. The runs are independent, so they can run in parallel. Restarts can't fix shape problems: on stretched clusters, the lowest-$J$ grouping is itself the wrong grouping, so keeping the lowest-$J$ run keeps a wrong answer.

**4. k-means++ picks seeds that are spread out.** The first seed is a data point chosen uniformly at random. Each next seed is chosen from the data points with probability
$$P(\text{pick } x^{(i)}) = \frac{D_i^2}{\sum_j D_j^2}$$
Points far from every seed chosen so far are much more likely to be picked. Squared distance is used because each point's $D_i^2$ is exactly what it would add to the cost if you stopped now and used the chosen seeds as centroids, so k-means++ picks seeds where the cost is. Code invariant: `D2[i]` always holds point $i$'s squared distance to its nearest chosen seed, and after each new seed it is updated with `np.minimum(D2, distance_to_new_seed)`. The whole seeding costs about as much as one k-means iteration. Chosen points and their exact duplicates have $D_i^2 = 0$, so they can never be picked again. Guard: if `D2.sum() == 0` before all $K$ seeds are chosen, the data has fewer than $K$ distinct points, and the probabilities would divide by zero.

**5. k-means++ works because uncovered blobs carry most of the weight, but it weakens as $K$ grows.** On our data, a point in a covered blob has $D^2 \approx 4$ on average ($w_c$), and a point in an uncovered blob has $D^2 \approx 104$ ($w_u$). So each point in an uncovered blob is about 26 times as likely to be picked. A miss is a pick that lands in an already-covered blob, and its chance at pick $j$ is approximately
$$\text{miss chance at pick } j = \frac{(j-1)\,w_c}{(j-1)\,w_c + (K-j+1)\,w_u}$$
This is the covered blobs' total weight divided by the total weight of all blobs. The formula rests on two assumptions. First, every earlier pick hit, so the $j-1$ seeds already chosen sit in $j-1$ different blobs and $K-j+1$ blobs are still uncovered. Second, the blobs are equal in size: the number of points per blob appears in every term and cancels, which is why it's missing from the formula. With unequal blobs, each blob's weight would need its own point count. Also, $w_c$ and $w_u$ are averages, so the formula gives an approximate miss chance, not an exact one.

As more blobs get covered, their many small weights add up, so misses become more likely. At $K = 16$, the miss chance rises from 0.26% at the second pick to 37% at the last. Every pick must hit, so the chance of success is the product of all the hit chances. Measured success: 96%, 71% and 36% for $K = 4, 9, 16$.

**6. `scikit-learn` runs greedy k-means++, which tries several candidates for each seed.** For each new seed, it draws $2 + \lfloor \ln K \rfloor$ candidates by $D^2$ sampling and keeps the one that makes $\sum D^2$ smallest. A candidate from an already-covered blob barely lowers $\sum D^2$, because the points in the uncovered blobs stay far from every seed, so it loses to a candidate from an uncovered blob. Measured success: 100%, 99.6% and 97% for $K = 4, 9, 16$. The cost is one extra pass over the data per candidate.

**7. Plain k-means++ guarantees the cost is not too far from the best.** On any dataset,
$$\mathbb{E}\bigl[J_{\text{after seeding}}\bigr] \le 8\,(\ln K + 2)\,J_{\text{best}}$$
Here $J_{\text{best}}$ is the lowest $J$ any grouping can achieve, and $\mathbb{E}$ is the average over the random choice of seeds. In words: on average, the seeds alone give a cost at most $8(\ln K + 2)$ times the best possible cost, which is about 38 times for $K = 16$. The k-means iterations can only lower $J$ further, because neither the assign step nor the update step increases $J$. It's a guarantee about the cost, not about whether the clusters look right: on stretched clusters, $J_{\text{best}}$ itself can be the wrong grouping.

**8. k-means++ has limits with outliers and with very large data.** An outlier is far from everything, so its $D^2$ is huge and it is likely to get its own seed, which spends a whole cluster on one point. More generally, $J$ gives far points a lot of weight because the distances are squared, so outliers pull the clusters toward them. Clean outliers out first. On very large data spread over many machines, picking seeds one at a time means one pass over all the data per seed, which is slow. k-means|| ("k-means parallel") picks many candidates in each pass, so it needs only a few passes, and it is the default in Spark MLlib.

**9. The course's method and today's library default differ.** The course uses random data points as seeds plus 50 to 1000 restarts. `scikit-learn`'s `KMeans` defaults to greedy k-means++ (`init="k-means++"`) with `n_init="auto"`, which means 1 run with k-means++, or 10 runs with `init="random"`.

### k-means - 4. Choosing the number of clusters $K$

**Symbols**

- $J^*(K)$: the lowest possible cost with $K$ clusters.
- $a(i)$: point $i$'s average distance to the other points in its own cluster.
- $b(i)$: point $i$'s average distance to the points of the nearest other cluster.
- $s(i)$: point $i$'s silhouette value.
- $\varepsilon$ (epsilon): a distance radius, used by DBSCAN and HDBSCAN.

**1. The lowest $J$ can't choose $K$.** Adding a cluster never makes the best possible cost worse: $J^*(K+1) \le J^*(K)$. Take the best clustering with $K$ centroids and put one more centroid on any data point that doesn't already have one. That point's distance to its centroid drops to zero, and every other point can still use its old centroid, so the cost can't go up. With one centroid per point, $J = 0$. So choosing $K$ by the lowest $J$ would always pick the largest $K$ you try.

**2. The course's elbow method looks for where adding clusters stops helping much.** Plot the lowest $J$ found against $K$, and pick the $K$ where the curve changes from falling steeply to falling slowly. Before that point, each new centroid takes over a whole true group, so $J$ drops a lot. After it, each new centroid only splits a group that already has a centroid, so $J$ drops a little. The method gives a clear answer only when the groups are well separated and the gaps between them are about the same size. When small groups sit inside larger groups, the curve shows two bends and can't say which one you want. When the data has no groups at all, the curve falls smoothly and has no elbow.

**3. Automatic elbow finders make the choice repeatable but keep the method's weaknesses.** Kneedle (in the `kneed` package) rescales both axes to the range 0 to 1, draws a straight line from the first point of the curve to the last, and picks the $K$ where the curve sits furthest below that line. `yellowbrick`'s `KElbowVisualizer` runs the whole sweep over $K$ and marks that point. They can still return a $K$ when the data has no groups, because a smooth, gently bending curve still has some point that is furthest from the line.

**4. The course's main advice is to choose $K$ from what the clusters will be used for.** Examples: how many T-shirt sizes are worth making, how many colours an image palette needs (the course lab uses 16), or how many clusters an IVF search index in `faiss` should have, which trades search speed against accuracy. Here k-means divides the data into a chosen number of pieces, and the data doesn't need to have natural groups at all.

**5. The silhouette score compares each point's closeness to its own cluster with its closeness to the next cluster.**
$$s(i) = \frac{b(i) - a(i)}{\max\bigl(a(i),\, b(i)\bigr)}$$
The value runs from $-1$ to $1$. Near $1$, the point is much closer to its own cluster than to the next one. Near $0$, it sits on the border between the two. Below $0$, it is closer to the other cluster than to its own. Average $s(i)$ over all points and pick the $K$ with the highest average. Unlike $J$, the silhouette can fall as $K$ grows: splitting a true group puts the two halves right next to each other, which makes $b(i)$ small for their points.

**6. The silhouette's value also says whether the groups are real.** Rough guide for the average: above 0.7, strong structure; 0.5 to 0.7, reasonable; 0.25 to 0.5, weak; below 0.25, no real structure. On data with no groups, every $K$ scored about 0.33, which warns that the clusters aren't real. It needs the distances between all pairs of points, which is slow on very large data, so `scikit-learn`'s `silhouette_score` has a `sample_size` option to use a random subset. Like $J$, it rewards compact, round groups. When stretched shapes dominate, as with two long parallel bars, it can score a wrong grouping higher than the true one.

**7. Other scores exist for comparing $K$ values.** Calinski–Harabasz divides the spread between the cluster centroids by the spread of points within their clusters; higher is better. Davies–Bouldin finds, for each cluster, the neighbouring cluster it overlaps with most, and averages those overlaps; lower is better. Both are fast and, like the silhouette, prefer round groups. The gap statistic compares $\log J$ on your data with $\log J$ on random data that has no groups, and picks the smallest $K$ after which adding one more cluster doesn't lower $\log J$ by more than random variation would.

**8. Density-based methods choose the number of clusters themselves.** DBSCAN calls a point a core point if at least `min_samples` points (counting itself) lie within distance $\varepsilon$ of it. Core points within $\varepsilon$ of each other are linked into the same cluster. A non-core point within $\varepsilon$ of a core point joins that core point's cluster as a border point. Every other point is labelled noise, with label $-1$. HDBSCAN removes the need to choose $\varepsilon$: it builds the clustering for every value of $\varepsilon$ at once, using a minimum spanning tree (the same structure as Kruskal's algorithm), and records how clusters merge as $\varepsilon$ grows. It then reads that history in reverse, as clusters splitting while $\varepsilon$ shrinks, and keeps the clusters that stay together over the widest range of $1/\varepsilon$. At each split, a piece counts as a cluster only if it has at least `min_cluster_size` points. On our data it found both long bars correctly and reported no clusters on the data with no groups. You still choose `min_cluster_size`, and the noise points labelled $-1$ must be removed before computing a silhouette score.

**9. The course's method and current practice differ.** The course uses the elbow method, or chooses $K$ from what the clusters will be used for. Current practice: choose $K$ from the later use when that use fixes it. Otherwise, compare $K$ values with the silhouette or a similar score, and read the score's value, not just which $K$ wins. Use HDBSCAN for stretched shapes, noisy data, or when you don't know how many groups to expect.
