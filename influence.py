"""
Module that handles influence function computation. Compatible with Tensorflow eager execution.
Adapted from repo: https://github.com/darkonhub/darkon/blob/master/darkon/influence/influence.py.
Based on work from Koh and Liang: https://github.com/kohpangwei/influence-release.
"""
import os
import numpy as np
import tensorflow as tf
from general_util import timing


class Influence:

    def __init__(self, train_loss_fn, test_loss_fn, model, data):
        """
        Influence object to inspect test losses.

        train_loss_fn : function
            Callable with parameters: model, x, y.
        test_loss_fn : function
            Callable with parameters: model, x, y.
        model : keras.Model
            Trained model.
        """
        self.train_loss_fn = train_loss_fn
        self.test_loss_fn = test_loss_fn
        self.model = model
        self.data = data
        self.num_total_train_instances = self.data.get_num_total_train_instances()
        self.trainable_variables = model.trainable_variables
        self.ihvp_config = {'scale': 1e4, 'damping': 0.01, 'num_repeats': 1,
                            'recursion_batch_size': 10, 'recursion_depth': 200}

    @timing
    def upweighting_influence(self, test_indices, train_indices=None, force_refresh=True):

        if train_indices is None:
            train_indices = self.data.get_all_train_indices()

        # compute test_grad_loss: gradient of the loss of test_indices
        test_grad = self._get_test_grad_loss(test_indices)
        test_loss_fn_no_args = self._get_test_loss_fn_no_args(test_indices)

        # compute inverse_hvp: use lissa to approximate the hvp between the hessian and test_grad_loss
        inverse_hvp = self._get_inverse_hvp_lissa(test_grad, test_loss_fn_no_args, force_refresh=force_refresh)

        # compute grad diffs for train_indices
        grad_diffs = self._get_grad_diffs(train_indices, inverse_hvp)

        return grad_diffs

    def _get_test_grad_loss(self, test_indices):
        """Compute the gradient of the test loss at test_indices."""

        x_test, y_test = self.data.get_test_instances(test_indices)
        test_grad_loss = self._get_grad_loss(self.test_loss_fn, x_test, y_test)
        return test_grad_loss

    def _get_train_grad_loss(self, train_indices):
        """Compute the gradient of the test loss at test_indices."""

        x_train, y_train = self.data.get_train_instances(train_indices)
        train_grad_loss = self._get_grad_loss(self.train_loss_fn, x_train, y_train)
        return train_grad_loss

    def _get_grad_loss(self, loss_fn, x, y):

        with tf.GradientTape() as tape:
            loss = self.test_loss_fn(self.model, x, y)

        grad_loss = tape.gradient(loss, self.trainable_variables)
        return grad_loss

    def _get_test_loss_fn_no_args(self, test_indices):
        x_test, y_test = self.data.get_test_instances(test_indices)
        test_loss_fn_no_args = self._get_loss_fn_no_args(self.test_loss_fn, x_test, y_test)
        return test_loss_fn_no_args

    def _get_loss_fn_no_args(self, loss_fn, x, y):

        def loss_fn_no_args():
            return loss_fn(self.model, x, y)

        return loss_fn_no_args

    @timing
    def _get_inverse_hvp_lissa(self, test_grad, test_loss_fn, force_refresh=True):
        inverse_hvp = None

        num_repeats = self.ihvp_config['num_repeats']
        recursion_depth = self.ihvp_config['recursion_depth']
        damping = self.ihvp_config['damping']
        scale = self.ihvp_config['scale']
        print_iter = recursion_depth / 10

        # return pre-computed inverse hvp
        ihvp_dir = '.ihvp'
        ihvp_path = os.path.join(ihvp_dir, 'inver_hvp.npy')
        if not force_refresh and os.path.exists(ihvp_path):
            inverse_hvp = np.load(ihvp_path)
            return inverse_hvp

        for i in range(num_repeats):
            cur_estimate = test_grad

            for j in range(recursion_depth):

                hessian_vector = self._get_hessian_vector_product(test_loss_fn, self.trainable_variables, cur_estimate)

                new_estimate = []
                for t_elem, c_elem, h_elem in zip(test_grad, cur_estimate, hessian_vector):
                    new_estimate.append(t_elem + (1 - damping) * c_elem - h_elem / scale)
                cur_estimate = new_estimate

                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    estimate_norm = np.linalg.norm(np.concatenate([tf.reshape(a, [-1]) for a in cur_estimate]))
                    print("Recursion at depth %s: norm is %.8lf" % (j, estimate_norm))

            if inverse_hvp is None:
                inverse_hvp = self._flatten(cur_estimate) / scale
            else:
                inverse_hvp += self._flatten(cur_estimate) / scale

        inverse_hvp /= num_repeats
        os.makedirs(ihvp_dir, exist_ok=True)
        np.save(ihvp_path, inverse_hvp)

        return inverse_hvp

    def _get_hessian_vector_product(self, f_fn, xs, v):

        if len(xs) != len(v):
            raise ValueError("xs and v must have the same length.")

        with tf.GradientTape(persistent=True) as tape:
            with tf.GradientTape(persistent=False) as tapetape:
                f = f_fn()

            grad = tapetape.gradient(f, xs)

            elem_prods = []
            for grad, v_elem in zip(grad, v):
                elem_prods.append(tf.multiply(grad, v_elem))

        hvp = tape.gradient(elem_prods, xs)
        return hvp

    @timing
    def _get_grad_diffs(self, train_indices, inverse_hvp):
        """Compute grad diffs for train_indices."""

        print_iter = self.num_total_train_instances / 100
        grad_diffs = np.zeros(self.num_total_train_instances)

        for i, train_ndx in enumerate(train_indices):

            # compute train_grad_loss: gradient of the loss of train_ndx
            train_grad = self._get_train_grad_loss(train_ndx)
            train_grad = self._flatten(train_grad)
            train_grad /= self.num_total_train_instances

            # return dot product between inverse_hvp and the train_grad_loss
            train_influence = tf.tensordot(inverse_hvp, train_grad, 1)
            grad_diffs[i] = train_influence

            if i % print_iter == 0:
                print('Training instance: %d' % i)

        return grad_diffs

    def _flatten(self, tensor_list):
        """Flattens list of tensors into a 1-D tensor."""
        return tf.constant(np.concatenate([tf.reshape(tensor, [-1]) for tensor in tensor_list]))
