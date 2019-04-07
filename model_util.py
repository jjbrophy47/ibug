"""
Model utility methods to: train, test, restore the model, etc.
"""
import os
import tensorflow as tf


def loss_fn(model, x, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model(x), labels=y))


def get_accuracy(model, x, y_true):
    logits = model(x)
    prediction = tf.argmax(logits, 1)
    equality = tf.equal(prediction, tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy


def get_optimizer():
    return tf.train.AdamOptimizer()


def test_single(model, x, y=None, debug=False):
    logits = model(x)
    prediction = tf.argmax(logits, 1)

    if debug:
        print(logits)
        print(prediction)


def restore(model, checkpoint_dir='.ckpt'):

    optimizer = get_optimizer()

    root = tf.train.Checkpoint(optimizer=optimizer, model=model)
    root.restore(tf.train.latest_checkpoint(checkpoint_dir))


def train(train_ds, model, checkpoint_dir='ckpt'):

    optimizer = get_optimizer()
    epochs = 10

    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    root = tf.train.Checkpoint(optimizer=optimizer, model=model)
    root.restore(tf.train.latest_checkpoint(checkpoint_dir))

    logdir = '.tb'
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    global_step = tf.train.get_or_create_global_step()

    for (batch, (images, labels)) in enumerate(train_ds):
        global_step.assign_add(1)

        with tf.GradientTape() as tape:
            loss = loss_fn(model, images, labels)

        # update weights
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

        # print update every 10 batches
        if batch % 10 == 0:
            acc = get_accuracy(model, images, labels).numpy()
            update_str = "Iteration {}, loss: {:.3f}, train accuracy: {:.2f}%"
            print(update_str.format(batch, loss_fn(model, images, labels).numpy(), acc * 100))
            root.save(checkpoint_prefix)

            with tf.contrib.summary.record_summaries_every_n_global_steps(1):
                tf.contrib.summary.scalar('accuracy', acc)

        # stop when epochs is reached
        if batch > epochs:
            break

    return model


def test(test_ds, model, print_intermediate=False):

    avg_acc = 0
    for (batch, (images, labels)) in enumerate(test_ds):
        avg_acc += get_accuracy(model, images, labels).numpy()

        if print_intermediate:
            if batch % 100 == 0 and batch != 0:
                print("Iteration {}, Average test accuracy: {:.2f}%".format(batch, (avg_acc/batch)*100))

    print("Final test accuracy: {:.2f}%".format(avg_acc/batch * 100))
