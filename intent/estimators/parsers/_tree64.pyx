# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Standardized tree.
"""
cimport cython

from cpython cimport Py_INCREF
from cpython.ref cimport PyObject

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf
from libc.math cimport pow

import numpy as np
cimport numpy as np
np.import_array()

# constants
cdef np_dtype_t = np.float64

cdef class _Tree64:

    property node_count_:
        def __get__(self):
            return self.node_count_

    property leaf_count_:
        def __get__(self):
            return self.leaf_count_

    def __cinit__(self,
                  SIZE_t[:]  children_left,
                  SIZE_t[:]  children_right,
                  SIZE_t[:]  feature,
                  DTYPE_t[:] threshold,
                  DTYPE_t[:] leaf_vals,
                  bint       lt_op):
        """
        Constructor.
        """
        self.children_left = children_left
        self.children_right = children_right
        self.feature = feature
        self.threshold = threshold
        self.leaf_vals = leaf_vals
        self.lt_op = lt_op

        self.node_count_ = 0
        self.leaf_count_ = 0
        self.root_ = self._add_node(0, 0, 0)

    def __dealloc__(self):
        """
        Destructor.
        """
        if self.root_:
            self._dealloc(self.root_)
            free(self.root_)

    cpdef np.ndarray predict(self, DTYPE_t[:, :] X):
        """
        Predict leaf values for x in X.
        """

        # In / out
        cdef SIZE_t n_samples = X.shape[0]
        cdef np.ndarray[DTYPE_t] out = np.zeros((n_samples,), dtype=np_dtype_t)

        # Incrementers
        cdef SIZE_t i = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root_

                while not node.is_leaf:

                    if self._test_threshold(X[i, node.feature], node.threshold):
                        node = node.left_child
                    else:
                        node = node.right_child

                out[i] = node.leaf_val

        return out

    cpdef np.ndarray apply(self, DTYPE_t[:, :] X):
        """
        Predict leaf index for x in X.
        """

        # In / out
        cdef SIZE_t n_samples = X.shape[0]
        cdef np.ndarray[int] out = np.zeros((n_samples,), dtype=np.int32)

        # Incrementers
        cdef SIZE_t i = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root_

                while not node.is_leaf:
                    if self._test_threshold(X[i, node.feature], node.threshold):
                        node = node.left_child
                    else:
                        node = node.right_child

                out[i] = node.leaf_id

        return out

    cpdef np.ndarray get_leaf_values(self):
        """
        Return 1d array of leaf values in order of their leaf IDs.
        """

        # result
        cdef DTYPE_t*   leaf_values = <DTYPE_t *>malloc(self.leaf_count_ * sizeof(DTYPE_t))
        cdef DTYPE_t[:] out = np.zeros((self.leaf_count_,), dtype=np_dtype_t)

        # incrementer
        cdef SIZE_t i = 0

        self._get_leaf_values(self.root_, leaf_values)

        # copy values to np.ndarray
        for i in range(self.leaf_count_):
            out[i] = leaf_values[i]

        # clean up
        free(leaf_values)

        return np.asarray(out)

    cpdef np.ndarray get_leaf_weights(self, DTYPE_t leaf_scale):
        """
        Return 1d array of leaf values in order of their leaf IDs.
        """

        # result
        cdef DTYPE_t* leaf_weights = <DTYPE_t *>malloc(self.leaf_count_ * sizeof(DTYPE_t))
        cdef DTYPE_t[:] out = np.zeros((self.leaf_count_,), dtype=np_dtype_t)

        # incrementer
        cdef SIZE_t i = 0

        self._get_leaf_weights(self.root_, leaf_weights, leaf_scale)

        # copy values to np.ndarray
        for i in range(self.leaf_count_):
            out[i] = leaf_weights[i]

        # clean up
        free(leaf_weights)

        return np.asarray(out)

    cpdef void update_node_count(self, DTYPE_t[:, :] X):
        """
        Increment each node count if x in X pass through it.
        """

        # In
        cdef SIZE_t n_samples = X.shape[0]

        # Incrementers
        cdef SIZE_t i = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root_

                while not node.is_leaf:
                    node.count += 1

                    if self._test_threshold(X[i, node.feature], node.threshold):
                        node = node.left_child
                    else:
                        node = node.right_child

                node.count += 1

    cpdef np.ndarray leaf_path(self, DTYPE_t[:, :] X, bint output, bint weighted):
        """
        Return 2d vector of leaf one-hot encodings, shape=(X.shape[0], no. leaves).
        """

        # In / out
        cdef SIZE_t        n_samples = X.shape[0]
        cdef SIZE_t        n_leaves = self.leaf_count_
        cdef DTYPE_t[:, :] out = np.zeros((n_samples, n_leaves), dtype=np_dtype_t)
        cdef DTYPE_t       val = 1.0

        # Incrementers
        cdef SIZE_t i = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root_

                while not node.is_leaf:
                    if self._test_threshold(X[i, node.feature], node.threshold):
                        node = node.left_child
                    else:
                        node = node.right_child

                val = 1.0

                if output:
                    val = node.leaf_val

                if weighted:
                    val /= node.count

                out[i][node.leaf_id] = val

        return np.asarray(out)

    cpdef np.ndarray feature_path(self, DTYPE_t[:, :] X, bint output, bint weighted):
        """
        Return 2d vector of feature one-hot encodings, shape=(X.shape[0], no. nodes).
        """

        # In / out
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_nodes = self.node_count_
        cdef DTYPE_t[:, :] out = np.zeros((n_samples, n_nodes), dtype=np_dtype_t)
        cdef DTYPE_t val = 1.0

        # Incrementers
        cdef SIZE_t i = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root_

                while not node.is_leaf:
                    val = 1.0

                    if weighted:
                        val /= node.count

                    out[i][node.node_id] = val

                    # traverse
                    if self._test_threshold(X[i, node.feature], node.threshold):
                        node = node.left_child
                    else:
                        node = node.right_child

                # leaf
                val = 1.0

                if output:
                    val = node.leaf_val

                if weighted:
                    val /= node.count

                out[i][node.node_id] = val

        return np.asarray(out)

    cpdef str tree_str(self):
        """
        Return string representation of the tree.
        """
        return self._tree_str(self.root_, '')

    # private
    cdef Node* _add_node(self,
                         SIZE_t node_id,
                         SIZE_t depth,
                         bint   is_left) nogil:
        """
        Pre-order traversal: Recursively create a subtree and return it.
        """
        cdef Node* node = self._initialize_node(node_id, depth, is_left)

        # decision node
        if self.children_left[node_id] != self.children_right[node_id]:
            node.feature = self.feature[node_id]
            node.threshold = self.threshold[node_id]

            if self.children_left[node_id] != -1:
                node.left_child = self._add_node(self.children_left[node_id], depth + 1, 0)

            if self.children_right[node_id] != -1:
                node.right_child = self._add_node(self.children_right[node_id], depth + 1, 1)

        # leaf node
        else:
            node.leaf_id = self.leaf_count_
            node.leaf_val = self.leaf_vals[node_id]
            node.is_leaf = 1
            self.leaf_count_ += 1

        self.node_count_ += 1

        return node

    cdef Node* _initialize_node(self,
                                SIZE_t node_id,
                                SIZE_t depth,
                                bint   is_left) nogil:
        """
        Create and initialize a new node.
        """
        cdef Node *node = <Node *>malloc(sizeof(Node))
        node.node_id = node_id
        node.leaf_id = -1
        node.count = 0
        node.depth = depth
        node.is_left = is_left
        node.is_leaf = 0
        node.feature = -1
        node.threshold = -1
        node.leaf_val = -1
        node.left_child = NULL
        node.right_child = NULL
        return node

    cdef void _get_leaf_values(self,
                               Node* node,
                               DTYPE_t* leaf_values) nogil:
        """
        Recursively fill 1d array of leaf values in order of their leaf IDs.
        """

        if node.is_leaf:
            leaf_values[node.leaf_id] = node.leaf_val

        else:
            self._get_leaf_values(node.left_child, leaf_values)
            self._get_leaf_values(node.right_child, leaf_values)

        return

    cdef void _get_leaf_weights(self,
                                Node* node,
                                DTYPE_t* leaf_weights,
                                DTYPE_t leaf_scale) nogil:
        """
        Recursively fill 1d array of leaf weights in order of their leaf IDs.
        """

        if node.is_leaf:
            leaf_weights[node.leaf_id] = pow(node.count, leaf_scale)

        else:
            self._get_leaf_weights(node.left_child, leaf_weights, leaf_scale)
            self._get_leaf_weights(node.right_child, leaf_weights, leaf_scale)

        return

    cdef bint _test_threshold(self, DTYPE_t fvalue, DTYPE_t threshold) nogil:
        """
        Test the feature value on the given threshold.
        """
        cdef bint result = 0

        if self.lt_op:
            result = 1 if fvalue < threshold else 0

        else:
            result = 1 if fvalue <= threshold else 0

        return result

    cdef void _dealloc(self, Node *node) nogil:
        """
        Recursively free all nodes in the subtree.

        NOTE: Does not deallocate "root" node, that must
              be done by the caller!
        """
        if not node:
            return

        # traverse to the bottom nodes first
        self._dealloc(node.left_child)
        self._dealloc(node.right_child)

        # free children
        free(node.left_child)
        free(node.right_child)

        # reset node properties just in case
        node.node_id = -1
        node.leaf_id = -1
        node.count = -1
        node.depth = -1
        node.is_left = 0
        node.is_leaf = 0
        node.feature = -1
        node.threshold = -1
        node.leaf_val = -1
        node.left_child = NULL
        node.right_child = NULL

    cdef str _tree_str(self, Node* node, str s):
        """
        Return string representation of the tree.

        Note
            - Uses the GIL, so use this ONLY for debugging.
        """
        if node == NULL:
            return ''

        space = node.depth * '\t'
        
        if node.is_leaf:
            s = f'\n{space}[Node {node.node_id}, leaf {node.leaf_id}] leaf_val: {node.leaf_val:.32f}'

        else:
            s = f'\n{space}[Node {node.node_id}] feature: {node.feature}, threshold: {node.threshold:.32f}'

            s += self._tree_str(node.left_child, s)
            s += self._tree_str(node.right_child, s)

        return s
