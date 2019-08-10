import tensorflow as tf


class dnn(object):
    def __init__(self,sentence_min_len,static_embeddings,embedding_size,hidden_size,learning_rate):
        self.sentence_min_len=sentence_min_len
        self.static_embeddings=static_embeddings
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.learning_rate=learning_rate

    def model_init(self):
        with tf.name_scope("dnn"):
            with tf.name_scope("placeholders"):
                inputs=tf.placeholder(dtype=tf.int32,shape=(None,self.sentence_min_len),name="inputs")
                targets=tf.placeholder(dtype=tf.float32,shape=(None,1),name="targets")
            with tf.name_scope("embeddings"):
                embedding_matrix=tf.Variable(initial_value=self.static_embeddings,trainable=False,name="embedding_matrix")
                embed=tf.nn.embedding_lookup(embedding_matrix,inputs,name="embed")
                #采用词向量相加得到句子向量（可改进）
                sum_embed=tf.reduce_sum(embed,axis=1,name="sum_embed")
            with tf.name_scope("model"):
                w1=tf.Variable(tf.random_normal(shape=(self.embedding_size,self.hidden_size),stddev=0.1),name="w1")
                b1=tf.Variable(tf.zeros(shape=(self.hidden_size),name="b1"))

                w2=tf.Variable(tf.random_normal(shape=(self.hidden_size,1),stddev=0.1),name="w2")
                b2=tf.Variable(tf.zeros(shape=(1),name="b2"))

                z1=tf.add(tf.matmul(sum_embed,w1),b1)
                a1=tf.nn.relu(z1)

                logits=tf.add(tf.matmul(a1,w2),b2)
                outputs=tf.nn.sigmoid(logits,name="outputs")
            with tf.name_scope("loss"):
                loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets,logits=logits))

            with tf.name_scope("optimizer"):
                optimizer=tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

            with tf.name_scope("evaluation"):
                correct_pred=tf.equal(tf.cast(tf.greater(outputs,0.5),tf.float32),targets)
                accuracy=tf.reduce_sum(tf.reduce_sum(tf.cast(correct_pred,tf.float32),axis=1))

