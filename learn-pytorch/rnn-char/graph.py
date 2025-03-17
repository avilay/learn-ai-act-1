import tensorflow as tf


class RnnGraph:
    def __init__(self, vocab_size, hyper_params):
        # inputs
        self.X = None
        self.Y = None

        # calculated tensors
        self.Y_hat = None
        self.initial_state = None
        self.final_state = None
        self.J = None
        self.optimizer = None

        # hyperparams
        n_seqs = hyper_params.n_seqs
        seq_len = hyper_params.seq_len
        lstm_size = hyper_params.lstm_size
        n_layers = hyper_params.n_layers
        dropout = hyper_params.dropout
        grad_clip = hyper_params.grad_clip
        learning_rate = hyper_params.learning_rate

        tf.reset_default_graph()
        self.X = tf.placeholder(tf.int32, [n_seqs, seq_len], name='X')
        self.Y = tf.placeholder(tf.int32, [n_seqs, seq_len], name='Y')

        # Build the network
        lstms = []
        for _ in range(n_layers):
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=(1 - dropout))
            lstms.append(drop)
        cell = tf.contrib.rnn.MultiRNNCell(lstms)
        self.initial_state = cell.zero_state(n_seqs, tf.float32)

        with tf.variable_scope('softmax'):
            W = tf.Variable(tf.truncated_normal((lstm_size, vocab_size), stddev=0.1))
            b = tf.Variable(tf.zeros(vocab_size))

        # Forward prop
        X_one_hot = tf.one_hot(self.X, vocab_size)  # shape = n_seqs, seq_len, vocab_size
        A_, self.final_state = tf.nn.dynamic_rnn(cell, X_one_hot,
                                                 initial_state=self.initial_state)  # shape = n_seqs, seq_len, lstm_size
        A = tf.reshape(A_, [-1, lstm_size])  # shape = n_seqs * seq_len, lstm_size
        Z = tf.add(tf.matmul(A, W), b)  # shape = n_seqs * seq_len, vocab_size
        self.Y_hat = tf.nn.softmax(Z, name='Y_hat')  # shape = n_seqs * seq_len, vocab_size

        # Cost
        Y_one_hot_ = tf.one_hot(self.Y, vocab_size)  # shape = n_seqs, seq_len, vocab_size
        Y_one_hot = tf.reshape(Y_one_hot_, Z.get_shape())  # shape = n_seqs * seq_len, vocab_size
        self.J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=Y_one_hot))

        # Optimizer
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.J, tvars), grad_clip)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).apply_gradients(zip(grads, tvars))
