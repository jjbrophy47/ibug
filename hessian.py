"""
Module to compute the hessian and hessian vector product.
Compatible with Tensorflow eager execution.
"""
import numpy as np
import tensorflow as tf


def hessian_vector_product(f_fn, xs, v):

    length = len(xs)
    if len(tf.reshape(v, [-1])) != length:
        raise ValueError("xs and v must have the same length.")

    with tf.GradientTape(persistent=True) as tape:
        with tf.GradientTape(persistent=False) as tapetape:
            f = f_fn()

        grad_list = tapetape.gradient(f, xs)
        print(grad_list)

        print(v)
        elem_prods = tf.multiply(grad_list, tf.reshape(v, [-1]))

        # elem_prods = []
        # for grad, v_elem in list(zip(grad_list, v)):
        #     print(grad, v_elem)
        #     elem_prods.append(tf.multiply(grad, v_elem))

    print(elem_prods)

    gradgrad_list = tape.gradient(elem_prods, xs)
    print(gradgrad_list)

    return gradgrad_list


def hessian(f_fn, xs):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xs)

        with tf.GradientTape(persistent=True) as tapetape:
            tapetape.watch(xs)

            f = f_fn()

        grad_list = tapetape.gradient(f, xs)
        print(grad_list)
        gradient = np.array(grad_list)
        print('gradient: {}'.format(gradient))

    h = []
    for grad in grad_list:
        print(grad[0])
        print(tape.gradient(grad[0], xs))
        h.append(tape.gradient(grad, xs))
    hessian = np.array(h)
    print('hessian:')
    print(hessian)


def testHessianVectorProduct():
    """Manually compute the Hessian explicitly for a low-dimensional problem
    and check that HessianVectorProduct matches multiplication by the explicit Hessian.
    Specifically, the Hessian of f(x) = x^T A x is H = A + A^T.
    We expect HessianVectorProduct(f(x), x, v) to be H v."""

    tf.enable_eager_execution()

    # n = 4
    # rng = np.random.RandomState([1, 2, 3])
    # m = rng.randn(n, n).astype("float32")

    # method 1: explicitly computing the hessian from the definition
    m = np.array([[1, 2], [2, 6]]).astype("float32")
    v = np.array([[8], [11]]).astype("float32")
    x = np.array([[1], [2]]).astype("float32")
    print(m)
    print(v)
    print(x)

    h = m + m.T
    print(h)

    hv = np.dot(h, v)
    print(hv)
    print()

    # method 2: explicilty compute the hessian using gradients
    mm = tf.constant(m)
    vv = tf.constant(v)

    x = tf.Variable(1.)
    y = tf.Variable(2.)

    p = [x, y]

    print(mm)
    print(vv)

    with tf.GradientTape(persistent=True) as tape:  # 1st derivative
        with tf.GradientTape(persistent=True) as tapetape:  # 2nd order derivatives

            xx = tf.reshape(tf.stack(p), (len(p), 1))
            f = tf.matmul(tf.transpose(xx), tf.matmul(mm, xx))

        grad_list = tapetape.gradient(f, [x, y])
        print(grad_list)

    h = []
    for grad in grad_list:
        gradgrad = tape.gradient(grad, [x, y])
        h.append(gradgrad)

    hessian = np.array(h)
    print(hessian)

    hv = np.dot(hessian, vv)
    print(hv)

    # method 3: implicitly compute the hessian, and multiply it by the given vector
    print()

    def f_fn():
        xx = tf.reshape(tf.stack(p), (len(p), 1))
        f = tf.matmul(tf.transpose(xx), tf.matmul(mm, xx))
        return f

    hv = hessian_vector_product(f_fn, [x, y], vv)
    print(hv)


if __name__ == '__main__':
    testHessianVectorProduct()
