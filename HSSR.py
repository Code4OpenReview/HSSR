import numpy as np
from model.AbstractRecommender import SeqAbstractRecommender
from util import DataIterator
import tensorflow as tf
from util import pad_sequences
import math
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from util import l2_loss


class HSSR(SeqAbstractRecommender):
    def __init__(self, sess, dataset, config):
        super(HSSR, self).__init__(dataset, config)
        self.L2 = config["L2"]
        self.hidden_size = config["hidden_size"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.max_seq_len = config["max_seq_len"]
        self.learning_rate = config["learning_rate"]

        self.lr_dc = config["lr_dc"]
        self.lr_dc_step = config["lr_dc_step"]
        self.lr = config["lr"]

        self.n_fold = config["n_fold"]

        self.cluster_size = config["cluster_size"]

        self.num_users, self.num_item = dataset.num_users, dataset.num_items
        self.user_pos_train = dataset.get_user_train_dict(by_time=True)

        self.train_seq = []
        self.train_tar = []
        for user, seqs in self.user_pos_train.items():
            for i in range(1, len(seqs)):
                self.train_seq.append(seqs[-i - self.max_seq_len:-i])
                self.train_tar.append(seqs[-i])

        self.train_seq_mask_padding, self.train_seq_padding = self._seq_process()
        self._build_items_transform_graph()
        self.sess = sess

    def _create_variable(self):
        self.batch_seqs_padding = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None])
        self.target_ph = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        stdv = 1.0 / math.sqrt(self.hidden_size)
        w_init = tf.random_uniform_initializer(-stdv, stdv)
        self.mask_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None])
        self.nasr_w1 = tf.get_variable('nasr_w1', [self.hidden_size, self.hidden_size], dtype=tf.float32, initializer=w_init)
        self.nasr_v = tf.get_variable('nasrv', [1, self.hidden_size], dtype=tf.float32, initializer=w_init)
        self.nasr_b = tf.get_variable('nasr_b', [self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.embedding = tf.get_variable(shape=[self.num_item, self.hidden_size], name='embedding', dtype=tf.float32, initializer=w_init)
        self.C = tf.get_variable('C', [2 * self.hidden_size, self.hidden_size], initializer=w_init)

    def _seq_process(self):
        mask = [[1] * len(items) for items in self.train_seq]
        mask = pad_sequences(mask, value=0, max_len = self.max_seq_len)
        seq_process = pad_sequences(self.train_seq, value=self.num_item, max_len = self.max_seq_len)
        return mask, seq_process

    def _build_items_transform_graph(self):
        adj = dok_matrix((self.num_item, self.num_item))
        for seq in self.user_pos_train.values():
            if len(seq) > 1:
                rows = seq[-self.max_seq_len:-1]
                cols = seq[-self.max_seq_len+1:]
                for row, col in zip(rows, cols):
                    adj[row, col] = adj[row, col] + 1
        # normalize by row
        self.norm_adj_mat = adj.multiply(csr_matrix(1 / (adj.sum(1) + 1e-5)))

    def build_graph(self):
        self._create_variable()
        Z_, S, X = self._h_gcn(self.norm_adj_mat, self.embedding)
        self.batch_seq_embedding = self._seq_em(self.embedding, S, X, self.batch_seqs_padding)
        self.all_logits = tf.matmul(self.batch_seq_embedding, self.embedding, transpose_b=True)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_ph, logits=self.all_logits)
        loss = tf.reduce_mean(loss)
        vars = tf.trainable_variables()
        lossL2 = [tf.nn.l2_loss(v) for v in vars if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']]
        self.loss_train = loss + self.L2 * tf.add_n(lossL2)
        global_step = tf.Variable(0)
        decay = self.lr_dc_step * len(self.train_seq) / self.batch_size
        learning_rate = tf.train.exponential_decay(self.lr, global_step=global_step, decay_steps=decay, decay_rate=self.lr_dc, staircase=True)
        self.train_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_train, global_step=global_step)

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        train_seq_len = [(idx, len(seq)) for idx, seq in enumerate(self.train_seq)]
        train_seq_len = sorted(train_seq_len, key=lambda x: x[1], reverse=True)
        train_seq_index, _ = list(zip(*train_seq_len))
        temp_loss = 100000000
        for epoch in range(self.epochs):
            total_loss = 0.0
            batch_loss = []
            flag = 0
            for bat_index in self._shuffle_index(train_seq_index):
                flag += 1
                item_seqs = [self.train_seq_padding[idx] for idx in bat_index]
                mask = [self.train_seq_mask_padding[idx] for idx in bat_index]
                bat_tars = [self.train_tar[idx] for idx in bat_index]
                feed = {self.target_ph: bat_tars,
                        self.mask_ph: mask,
                        self.batch_seqs_padding: item_seqs}  # b*L

                loss, _ = self.sess.run([self.loss_train, self.train_opt], feed_dict=feed)
                batch_loss.append(loss)
                total_loss += loss
            isVal = True
            self.logger.info("epoch %d\t%f:\t%s" % (epoch, total_loss, self.evaluate_model(isVal=isVal)))

            if abs(total_loss-temp_loss) < 0.0000001 or total_loss > temp_loss:
                break
            temp_loss = total_loss
        self.logger.info("test result: \t%s" % (self.evaluate_model(isVal=False)))

    def _shuffle_index(self, seq_index):
        index_chunks = DataIterator(seq_index, batch_size=self.batch_size * 32, shuffle=False,
                                    drop_last=False)
        index_chunks = list(index_chunks)
        index_chunks_iter = DataIterator(index_chunks, batch_size=1, shuffle=True, drop_last=False)
        for indexes in index_chunks_iter:
            indexes = indexes[0]
            indexes_iter = DataIterator(indexes, batch_size=self.batch_size, shuffle=True,
                                        drop_last=True)
            for bat_index in indexes_iter:
                yield bat_index

    def evaluate_model(self, isVal=True):
        return self.evaluator.evaluate(self, val=isVal)

    def predict(self, users, items):
        users = DataIterator(users, batch_size=self.batch_size, shuffle=False, drop_last=False)
        all_ratings = []
        for bat_user in users:
            cur_batch_size = len(bat_user)
            bat_items = [self.user_pos_train[user][-self.max_seq_len:] for user in bat_user]
            bat_items = pad_sequences(bat_items, value=self.num_item)
            mask = [[1] * len(self.user_pos_train[user]) for user in bat_user]
            mask = pad_sequences(mask, value=0)
            if cur_batch_size < self.batch_size:
                pad_size = self.batch_size - cur_batch_size
                bat_items = np.concatenate([bat_items, [bat_items[-1]] * pad_size], axis=0)
                mask = np.concatenate([mask, [mask[-1]] * pad_size], axis=0)
            feed = {self.batch_seqs_padding: bat_items, self.mask_ph:mask}  # b*L
            bat_ratings = self.sess.run(self.all_logits, feed_dict=feed)
            all_ratings.extend(bat_ratings[:cur_batch_size])
        all_ratings = np.array(all_ratings)
        if items is not None:
            all_ratings = [all_ratings[idx][u_item] for idx, u_item in enumerate(items)]

        return all_ratings

    def _h_gcn(self, A, X):
        layers_size_1 = self.hidden_size
        layers_size_2 = self.cluster_size
        tempA = A
        tempX = X
        S_all = []
        Z_ = self._gcn_embed(tempA, tempX, n_layers=1, layers_size=[layers_size_1])
        S = self._gcn_embed(tempA, tempX, n_layers=1, layers_size=[layers_size_2[0]])
        S = tf.nn.softmax(S, 1)
        S_all.append(S)
        tempX = tf.matmul(S, Z_, transpose_a=True)
        tempA = tf.matmul(S,tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(tempA),S),transpose_a=True)
        row_sum = tf.reduce_sum(tempA,1)
        D = tf.diag(tf.pow(row_sum,tf.constant([-0.5]*row_sum.shape[-1])))
        tempA = tf.matmul(tf.matmul(D, tempA), D)
        self.cluster_adj = tempA
        for i in range(1,len(layers_size_2)):
            Z = self._gcn___(tempA, tempX, n_layers=1, layers_size=[layers_size_1])
            S = self._gcn___(tempA, tempX, n_layers=1, layers_size=[layers_size_2[i]])
            S = tf.nn.softmax(S, 1)
            tempX = tf.matmul(S, Z, transpose_a=True)
            tempA = tf.matmul(tf.matmul(S, tempA, transpose_a=True), S)
            row_sum = tf.reduce_sum(tempA, 1)
            D = tf.diag(tf.pow(row_sum, tf.constant([-0.5] * row_sum.shape[-1])))
            tempA = tf.matmul(tf.matmul(D, tempA), D)
            self.cluster_adj = tempA
            S_all.append(S)
        X = tempX
        S = S_all[0]
        for i in range(1,len(S_all)):
            S = tf.matmul(S, S_all[i])
        if (len(S_all) > 1):
            S = tf.nn.softmax(S, 1)
        return Z_, S, X

    def _seq_em(self, Z, S, X, batch_seqs_padding):

        seqs_em = []
        for i in range(self.batch_size):

            seq_len = tf.to_int32(tf.reduce_sum(self.mask_ph[i], -1))
            seq = batch_seqs_padding[i][:seq_len]
            _S = tf.nn.embedding_lookup(S, seq)

            last_cluster = tf.expand_dims(tf.nn.embedding_lookup(S, seq[-1]),0)
            last_clEm = tf.matmul(last_cluster, X)
            seq_em_all = tf.expand_dims(tf.reduce_sum(tf.nn.embedding_lookup(Z, seq), 0),0)

            sum_S = tf.nn.softmax(tf.reduce_sum(_S, 0))
            seq_em_cluster = tf.matmul(tf.reshape(sum_S, [1, -1]), X)
            seq_em_last = tf.expand_dims(tf.nn.embedding_lookup(Z, seq[-1]),0)

            # -------------------default--------------------------
            seq_em = tf.concat([last_clEm, seq_em_cluster, seq_em_all, seq_em_last], 0)
            weights = tf.matmul(tf.nn.sigmoid((tf.matmul(seq_em, self.nasr_w1) + self.nasr_b)), self.nasr_v, transpose_b=True)
            seq_em = tf.reduce_sum(tf.nn.softmax(weights) * seq_em, 0)
            # -------------------------------------------------------------

            # -------------------Ablation Analysis 1----------------------
            # seq_em = tf.concat([last_clEm, seq_em_cluster, seq_em_all, seq_em_last], 0)
            # seq_em = tf.reduce_mean(seq_em, 0)
            # -------------------------------------------------------------

            # -------------------Ablation Analysis 4--------------------------
            # seq_em = tf.concat([seq_em_all, seq_em_last], 0)
            # weights = tf.matmul(tf.nn.sigmoid((tf.matmul(seq_em, self.nasr_w1) + self.nasr_b)), self.nasr_v,
            #                     transpose_b=True)
            # seq_em = tf.reduce_sum(tf.nn.softmax(weights) * seq_em, 0)  # 1*d
            # seq_em = tf.matmul(tf.concat([tf.expand_dims(seq_em, 0), seq_em_last], -1), self.C)
            # seq_em = tf.squeeze(seq_em, 0)  # d
            # -------------------------------------------------------------

            # -------------------Ablation Analysis 5--------------------------
            # seq_em = tf.concat([seq_em_all, seq_em_cluster], 0)
            # weights = tf.matmul(tf.nn.sigmoid((tf.matmul(seq_em, self.nasr_w1) + self.nasr_b)), self.nasr_v,
            #                     transpose_b=True)
            # seq_em = tf.reduce_sum(tf.nn.softmax(weights) * seq_em, 0)  # 1*d
            # -------------------------------------------------------------

            # -------------------Ablation Analysis 6--------------------------
            # seq_em = tf.concat([seq_em_last, last_clEm], 0)
            # weights = tf.matmul(tf.nn.sigmoid((tf.matmul(seq_em, self.nasr_w1) + self.nasr_b)), self.nasr_v,
            #                     transpose_b=True)
            # seq_em = tf.reduce_sum(tf.nn.softmax(weights) * seq_em, 0)  # 1*d
            # -------------------------------------------------------------
            seqs_em.append(seq_em)
        return seqs_em

    def _gcn_embed(self, A, X, n_layers=3, layers_size=[64, 64, 64]):
        n_fold = self.n_fold
        if len(layers_size) != n_layers:
            raise Exception("number of layer is inconsistent with the defined")
        A_fold_hat = self._split_A_hat(A)
        embeddings = X
        for layer in range(n_layers):
            temp_embed = []
            for f in range(n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))  # A * X

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.layers.dense(inputs=embeddings, units=layers_size[layer], activation=tf.nn.leaky_relu)
        return embeddings

    """
    A: k * k
    X: k * d
    """
    def _gcn___(self, A, X, n_layers=3, layers_size=[64,64,64]):
        if len(layers_size) != n_layers:
            raise Exception("number of layer is inconsistent with the defined")
        embeddings = X
        for layer in range(n_layers):
            embeddings = tf.layers.dense(inputs=tf.matmul(A, embeddings), units=layers_size[layer],activation=tf.nn.leaky_relu)
        return embeddings

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _split_A_hat(self, X):
        n_fold = self.n_fold
        A_fold_hat = []
        fold_len = self.num_item // n_fold
        for i_fold in range(n_fold):
            start = i_fold * fold_len
            if i_fold == n_fold - 1:
                end = self.num_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat
