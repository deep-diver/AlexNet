import tensorflow as tf

# activation function is not specified yet
def single_gpu_convnet(input):
    conv1 = tf.nn.conv2d(input, [11, 11, 3, 96], [1, 4, 4, 1], "SAME")
    lrn1 = tf.nn.local_response_normalization(conv1)
    pool1 = tf.nn.max_pool(lrn1)

    conv2 = tf.nn.conv2d(pool1, [5, 5, 48, 256], [1, 4, 4, 1], "SAME")
    lrn2 = tf.nn.local_response_normalization(conv2)
    pool2 = tf.nn.max_pool(lrn2)

    conv3 = tf.nn.conv2d(pool2, [3, 3, 128, 384], [1, 4, 4, 1], "SAME")
    conv4 = tf.nn.conv2d(conv3, [3, 3, 192, 384], [1, 4, 4, 1], "SAME")

    conv5 = tf.nn.conv2d(conv4, [5, 5, 192, 256], [1, 4, 4, 1], "SAME")
    lrn5 = tf.nn.local_response_normalization(conv5)
    pool5 = tf.nn.max_pool(lrn5)

    flat = tf.contrib.layers.flatten(pool5)

    fcl1 = tf.contrib.layers.fully_connected(flat, 4096)
    fcl2 = tf.contrib.layers.fully_connected(fcl1, 4096)
    out = tf.contrib.layers.fully_connected(fcl2, 1000, activation_fn=None)

    return out

def multi_gpu_convnet():
    # on GPU #1
    with tf.device('/gpu:0'):
        # 1st Convolutional Layer
        conv1_1 = tf.nn.conv2d(input, [11, 11, 3, 48], [1, 4, 4, 1], "SAME")
        lrn1_1 = tf.nn.local_response_normalization(conv1_1)
        pool1_1 = tf.nn.max_pool(lrn1_1)

        # 2nd Convolutional Layer
        conv2_1 = tf.nn.conv2d(pool1_1, [5, 5, 48, 128], [1, 4, 4, 1], "SAME")
        lrn2_1 = tf.nn.local_response_normalization(conv2_1)
        pool2_1 = tf.nn.max_pool(lrn2_1)

        conv3_1 = tf.nn.conv2d(pool2_1, [3, 3, 128, 192], [1, 4, 4, 1], "SAME")


    # on GPU #2
    with tf.device('/gpu:1'):
        # 1st Convolutional Layer
        conv1_2 = tf.nn.conv2d(input, [11, 11, 3, 48], [1, 4, 4, 1], "SAME")
        lrn1_2 = tf.nn.local_response_normalization(conv1_2)
        pool1_2 = tf.nn.max_pool(lrn1_2)

        # 2nd Convolutional Layer
        conv2_2 = tf.nn.conv2d(pool1_2, [5, 5, 48, 128], [1, 4, 4, 1], "SAME")
        lrn2_2 = tf.nn.local_response_normalization(conv2_2)
        pool2_2 = tf.nn.max_pool(lrn2_2)

        conv3_2 = tf.nn.conv2d(pool2_2, [3, 3, 128, 192], [1, 4, 4, 1], "SAME")

    ############ 3rd Convolutional Layer #########################################
    ##############################################################################
    ############ 4th Convolutional Layer #########################################

    with tf.device('/gpu:0'):
        conv4_1_input = tf.concat([conv3_1, conv3_2], 0)
        conv4_1 = tf.nn.conv2d(conv4_1_input, [3, 3, 192, 192], [1, 4, 4, 1], "SAME")

        conv5_1 = tf.nn.conv2d(conv4_1, [5, 5, 192, 128], [1, 4, 4, 1], "SAME")
        lrn5_1 = tf.nn.local_response_normalization(conv5_1)
        pool5_1 = tf.nn.max_pool(lrn5_1)

        flat_1 = tf.contrib.layers.flatten(pool5_1)

    with tf.device('/gpu:1'):
        conv4_2_input = tf.concat([conv3_2, conv3_1], 0)
        conv4_2 = tf.nn.conv2d(conv4_2_input, [3, 3, 192, 192], [1, 4, 4, 1], "SAME")

        conv5_2 = tf.nn.conv2d(conv4_2, [5, 5, 192, 128], [1, 4, 4, 1], "SAME")
        lrn5_2 = tf.nn.local_response_normalization(conv5_2)
        pool5_2 = tf.nn.max_pool(lrn5_2)

        flat_2 = tf.contrib.layers.flatten(pool5_2)

    with tf.device('/gpu:0'):
        fcl1_1_input = tf.concat([flat_1, flat_2], 0)
        fcl1_1 = tf.contrib.layers.fully_connected(fcl1_1_input, 2048)

    with tf.device('/gpu:1'):
        fcl1_2_input = tf.concat([flat_1, flat_2], 0)
        fcl1_2 = tf.contrib.layers.fully_connected(fcl1_2_input, 2048)

    with tf.device('/gpu:0'):
        fcl2_1_input = tf.concat([fcl1_1, fcl1_2], 0)
        fcl2_1 = tf.contrib.layers.fully_connected(fcl2_1_input, 2048)

    with tf.device('/gpu:1'):
        fcl2_2_input = tf.concat([fcl1_1, fcl1_2], 0)
        fcl2_2 = tf.contrib.layers.fully_connected(fcl2_2_input, 2048)

    with tf.device('/gpu:0'):
        fcl3_1_input = tf.concat([fcl2_1, fcl2_2], 0)
        out = tf.contrib.layers.fully_connected(fcl3_1_input, 1000, activation_fn=None)

    return out

def main():
    # hyper-parameters
    leraning_rate = 0.1

    input = tf.placeholder(tf.float32, [None, 224, 224, 3])
    label = tf.placeholder(tf.int32, [None, 1000])

    logits = single_gpu_convnet()

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(optimizer,
                    feed_dict={
                        input: None,
                        label: None,
                    })

if __name__ == "__main__":
    main()
