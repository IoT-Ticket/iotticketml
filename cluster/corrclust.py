import numpy as np

    
def _get_centroids(X, cluster_indices):
    means = np.empty((len(cluster_indices), X.shape[1]))
    stds = np.empty((len(cluster_indices), X.shape[1]))
    for i in range(means.shape[0]):
        means[i, :] = X[cluster_indices[i], :].mean(0)
        stds[i, :] = X[cluster_indices[i], :].std(0)
    return means, stds
    
    
def _unique(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item
        
        
def _unique_count(lst):
    last = lst[0]
    count = 0
    for item in lst:
        if item == last:
            count += 1
            continue
        yield count
        last = item
        count = 1
    yield count
        

def _sort_and_deduplicate(l):
    sl = sorted(l, reverse=True)
    return list(reversed(list(_unique(sl)))), list(reversed(list(_unique_count(sl))))


def _cluster_indices(lst, cluster):
    count = -1
    for item in lst:
        count += 1
        if item == cluster:
            yield count
            
            
def _indices_by_cluster(lst, clusters):
    for cl in clusters:
        yield list(_cluster_indices(lst, cl))
        

def _deepen(a):
    if len(a) == 1:
        return a
    else:
        return [a[0], [_deepen(a[1:])]]
    
    
def _arrmerge(a, b):
    
    if (len(a[-1]) == 1 or len(b[0]) == 1):
        if a[-1][0] == b[-1][0]:
            a[-1] = [a[-1][0], b[0][-1]]
        else:
            a.append(b[0])
        return a
    if a[-1][0] != b[0][0]:
        a.append(b[0])
        return a
    else:
        if len(a[-1]) == 1:
            return [a[-1][0], _arrmerge(a[-1][-1], b[-1][-1])]
        else:
            return a[:-1] + [[a[-1][0], _arrmerge(a[-1][-1], b[-1][-1])]]
    
    
def _mergearrs(arrs):
    res = [arrs[0]]
    for i in range(1, len(arrs)):
        if res[-1][0] == arrs[i][0]:
            if len(res[-1]) == 1:
                res[-1] = [res[-1][0], arrs[i][-1]]
            else:
                res[-1] = [res[-1][0], _arrmerge(res[-1][-1], arrs[i][-1])]
        else:
            res.append(arrs[i])
    return res


def _node_content(lab, child):
    return '[.{{{}}} {}]'.format(lab, child)


def _tikz_tree(arrs):
    res = ''
    for i in arrs:
        if len(i) == 1:
            res += _node_content(i[0], '')
        else:
            res += _node_content(i[0], _tikz_tree(i[-1]))
    return res


def _labels_to_tikz(arrs):
    return '\Tree[.{{data}} {}]'.format(_tikz_tree(arrs))
    
    
def _insert_count(lab, tree, cnt):
    for i in tree:
        if (len(lab) == 1 and i[0] == lab[0]):
            i[0] = i[0] + '\\\\(' + str(cnt) + ')'
            return tree
        elif (i[0] == lab[0] or i[0][0] == lab[0] or i[0][:2] == lab[0] or i[0][:3] == lab[0]):
            if len(i) == 1:
                i[0] = i[0] + '}\\\\(' + str(cnt) + ')'
                return tree
            else:
                i = _insert_count(lab[1:], i[-1], cnt)
                break
    return tree
    
    
class _CorrClusBase(object):
    """Base class for CHUNX and CRUSHES correlation clustering algorithms.
    
    Author: ilari.kampman@wapice.com
    """
    def __init__(self):
        pass
    
    
    def fit(self, X):
        assert(np.all(~np.isnan(X)))
        assert(np.all(~np.isinf(X)))
        # Center data
        centered = X - X.mean(axis=0)
        # Normalize data
        for i in range(X.shape[0]):
            centered[i, :] = centered[i, :]/np.linalg.norm(centered[i, :], 2)
        # Compute eigenvectors
        _, V = np.linalg.eig(np.cov(centered, rowvar=False))
        self.X_ = centered
        cs = np.dot(centered, np.concatenate((V, -V), axis=1))
        self.XV = cs
        self.cord = np.argsort(-cs, axis=1)
        return self
    
    
    def get_cluster_index(self, index):
        """Get the cluster label of a particular data entry of the original data set.
        
        Parameters
        ----------
        
        index : int
            The index of the data entry in the original data set.
            
        Returns
        -------
        
        label : int
            The label of the cluster where the data entry is clustered into.
        """
        for lst in self.clusters_:
            if self.labels_[index] == lst:
                return self.clusters_.index(lst)
        return None
        
    
    def get_tikz_tree(self, min_size=10):    
        """Get cluster tree in LaTeX TIKZ format
        
        Parameters
        ----------
        
        min_size : int
            The minimum size of a cluster displayed in the tree.
            
        Returns
        -------
        
        tikz_tree : str
            The cluster tree in TIKZ format.
        """
        filtered_is = np.where(np.array(self.cluster_counts_) > min_size)[0]
        tree_clusters = [self.clusters_[i] for i in filtered_is]
        a = [['+' + str(j+1) if j < self.X_.shape[1] else 
              '-' + str(j - self.X_.shape[1] + 1) for j in i] for i in tree_clusters]
        ca = list(np.array(self.cluster_counts_)[filtered_is])
        da = [_deepen(i) for i in a]
        my_tree = _mergearrs(da)
        for i in range(len(ca)):
            mod_tree = _insert_count(a[i], my_tree, ca[i])
        return _labels_to_tikz(mod_tree)
    
    
def _formclusters(sortedargs, max_depth=None, n_relax=1):
    """CHUNX-PARTITION: Recursive algorithm for returning labeled data from the argument
    sorted array of significance factors with every eigenvector for every data entry.

    Parameters
    ----------
        
    sortedargs : array-like, shape=(n_samples, n_singularcomponents),  
        the argument sorted array of significance factors with every singular
        vector for every data entry.
            
    max_depth : int, default=None
        The maximum tree depth
            
    n_relax : int, default=1
        The number of elements in a leaf which is considered as a satisfiable
        cluster size. Default value goes through all the singular components.
            
    Returns
    -------
        
    labels : list of lists
        A list of cluster labels in which the indices correspond the row
        indices of the original data.
    """
    labels = [[] for i in range(sortedargs.shape[0])]
    if max_depth is not None:
        max_depth -= 1
        if max_depth < 0:
            return labels
    maxargs = sortedargs[:, 0]
    uargs, counts = np.unique(maxargs, return_counts=True)
    for i in range(uargs.shape[0]):
        inds = np.where(maxargs == uargs[i])[0]
        for j in range(inds.shape[0]):
            labels[inds[j]].append(uargs[i])
        if counts[i] > n_relax:
            sublabels = _formclusters(sortedargs[inds, 1:],
                                      max_depth=max_depth,
                                      n_relax=n_relax)
            for k in range(inds.shape[0]):
                labels[inds[k]] += sublabels[k]
    return labels
    
    
class CHUNX(_CorrClusBase):
    """A class for CHUNX correlation clustering algorithm.

    Author: ilari.kampman@wapice.com
    """
    def __init__(self, max_components=None, max_proportion=0.50):
        self.max_components = max_components
        self.max_proportion = max_proportion
    
    
    def fit(self, X):
        """Compute hierarchical clusters by the input data X's singular componets.

        Parameters
        ----------
        
        X : array-like, shape=(n_samples, n_features),  
            Training data for computing generating the clusters. The data must not
            contain nan or inf values. The zero vectors will also cause problems
            with the clustering
        """
        super(CHUNX, self).fit(X)
        # Label entries by their most significant singular components
        self.labels_ = _formclusters(self.cord, max_depth=self.max_components,
                                     n_relax=np.floor(self.max_proportion*X.shape[0]))
        # Form the lists of unique clusters and cluster counts
        self.clusters_, self.cluster_counts_ = _sort_and_deduplicate(self.labels_)
        self.indices_in_clusters_ = list(_indices_by_cluster(self.labels_, self.clusters_))
        self.centroids_, self.centroid_stds_ = _get_centroids(X, self.indices_in_clusters_)
        
        return self
    
    
class CRUSHES(_CorrClusBase):
    """A class for CRUSHES correlation clustering algorithm.

    Author: ilari.kampman@wapice.com
    """
    def __init__(self, precision=0.50):
        self.precision = precision
    
    
    def fit(self, X):
        """Compute hierarchical clusters by the input data X's singular componets.

        Parameters
        ----------
        
        X : array-like, shape=(n_samples, n_features),  
            Training data for computing generating the clusters. The data must not
            contain nan or inf values. The zero vectors will also cause problems
            with the clustering
        """
        super(CRUSHES, self).fit(X)
        # This corresponds to correlation produced by each component
        Psquared = self.XV**2
        # Sorted
        Delta = -np.sort(-Psquared, axis=1)
        # Find and save component count in which each sample matches the precision
        self.comp_counts_ = np.zeros(Delta.shape[0], dtype=np.int)
        for i in range(len(self.comp_counts_)):
            cumcorr = 0.0
            compcount = 0
            while (compcount == 0 or np.sqrt(cumcorr) < self.precision):
                cumcorr += Delta[i, compcount] 
                compcount += 1
            self.comp_counts_[i] = compcount
        # Save the maximum number of components
        self.max_sig_comps_ = np.max(self.comp_counts_)
        # Form a list of cluster labels from the component counts
        self.labels_ = [list(self.cord[i, :self.comp_counts_[i]])
                        for i in range(len(self.comp_counts_))]
        # Find unique cluster labels and sample counts in every cluster
        self.clusters_, self.cluster_counts_ = _sort_and_deduplicate(self.labels_)
        # Save original indices of each sample in each cluster
        self.indices_in_clusters_ = list(_indices_by_cluster(self.labels_, self.clusters_))
        # Compute centroids and centroid standard deviations
        self.centroids_, self.centroid_stds_ = _get_centroids(X, self.indices_in_clusters_)
        return self
