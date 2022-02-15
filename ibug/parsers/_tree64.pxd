"""
64-bit-precision tree.
"""
import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DTYPE_t          # Type of X
ctypedef np.npy_intp    SIZE_t           # Type for indices and counters
ctypedef np.npy_int32   INT32_t          # Signed 32 bit integer
ctypedef np.npy_uint32  UINT32_t         # Unsigned 32 bit integer

"""
Node
"""
cdef struct Node:
    SIZE_t  node_id                  # Node identifier
    SIZE_t  leaf_id                  # Leaf identifier
    SIZE_t  count                    # Number of samples in the node
    SIZE_t  depth                    # Depth of node
    bint    is_left                  # Whether this node is a left child
    bint    is_leaf                  # Leaf node if True, otherwise decision node
    SIZE_t  feature                  # Feature index (decision node)
    DTYPE_t threshold                # Feature threshold (decision node)
    DTYPE_t leaf_val                 # Leaf value (leaf node)
    Node*   left_child               # Left child node pointer
    Node*   right_child              # Right child node pointer

"""
Structure to hold a SIZE_t pointer and the no. elements.
"""
cdef struct IntList:
    SIZE_t* arr
    SIZE_t  n

cdef class _Tree64:
    """
    The Tree object is a binary tree structure constructed by the
    TreeBuilder. The tree structure is used for predictions.
    """

    # Inner structures
    cdef SIZE_t[:]  children_left
    cdef SIZE_t[:]  children_right
    cdef SIZE_t[:]  feature
    cdef DTYPE_t[:] threshold
    cdef DTYPE_t[:] leaf_vals
    cdef bint       lt_op
    cdef Node*      root_
    cdef SIZE_t     node_count_
    cdef SIZE_t     leaf_count_

    # Python API
    cpdef np.ndarray predict(self, DTYPE_t[:, :] X)
    cpdef np.ndarray apply(self, DTYPE_t[:, :] X)
    cpdef np.ndarray get_leaf_values(self)
    cpdef np.ndarray get_leaf_weights(self, DTYPE_t leaf_scale)
    cpdef void       update_node_count(self, DTYPE_t[:, :] X)
    cpdef np.ndarray leaf_path(self, DTYPE_t[:, :] X, bint output, bint weighted)
    cpdef np.ndarray feature_path(self, DTYPE_t[:, :] X, bint output, bint weighted)
    cpdef str        tree_str(self)

    # C API
    cdef Node* _add_node(self, SIZE_t node_id, SIZE_t depth, bint is_left) nogil
    cdef Node* _initialize_node(self, SIZE_t node_id, SIZE_t depth, bint is_left) nogil
    cdef void  _get_leaf_values(self, Node* node, DTYPE_t* leaf_values) nogil
    cdef void  _get_leaf_weights(self, Node* node, DTYPE_t* leaf_weights, DTYPE_t leaf_scale) nogil
    cdef bint  _test_threshold(self, DTYPE_t fvalue, DTYPE_t threshold) nogil
    cdef void  _dealloc(self, Node *node) nogil
    cdef str   _tree_str(self, Node* node, str s)
