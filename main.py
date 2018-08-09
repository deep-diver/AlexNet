import tensorflow as tf

# activation function is not specified yet
def main():
    # hyper-parameters
    leraning_rate = 0.1
    
    input = tf.placeholder(tf.float32, [None, 224, 224, 3])

    conv1 = tf.nn.conv2d(input, [11, 11, 3, 96], [1, 4, 4, 1], "SAME")
    lrn1 = tf.nn.local_response_normalization(conv1)
    pool1 = tf.nn.max_pool(lrn1)

    conv2 = tf.nn.conv2d(input, [5, 5, 48, 256], [1, 4, 4, 1], "SAME")
    lrn2 = tf.nn.local_response_normalization(conv2)
    pool2 = tf.nn.max_pool(lrn2)

    conv3 = tf.nn.conv2d(input, [3, 3, 128, 384], [1, 4, 4, 1], "SAME")
    conv4 = tf.nn.conv2d(input, [3, 3, 192, 384], [1, 4, 4, 1], "SAME")
    conv5 = tf.nn.conv2d(input, [5, 5, 192, 256], [1, 4, 4, 1], "SAME")

    flat = tf.contrib.layers.flatten(conv5)

    fcl1 = tf.contrib.layers.fully_connected(flat, 4096)
    fcl2 = tf.contrib.layers.fully_connected(fcl1, 4096)
    out = tf.contrib.layers.fully_connected(fcl2, 1000, activation_fn=None)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(optimizer,
                    feed_dict={
                        input: None,
                        label: None,
                    })

if __name__ == "__main__":
    main()