
import tensorflow as tf

class TextCNN(object):
    def __init__(self,config,wordEmbedding):
        self.inputX=tf.placeholder(tf.int32,[None,config.sequenceLength],name="inputX")
        self.inputY=tf.placeholder(tf.int32,[None],name="inputY")

        self.dropoutKeepProb=tf.placeholder(tf.float32,name="dropoutKeepProb")
        l2Loss=tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.w=tf.Variable(tf.cast(wordEmbedding,dtype=tf.float32,name="word2vec"),name="W")
            self.embeddedWords=tf.nn.embedding_lookup(self.w,self.inputX)

            self.embeddedWordsExpanded=tf.expand_dims(self.embeddedWords,-1)

        pooledOutputs=[]
        for i ,filterSize in enumerate(config.model.filterSize):
            with tf.name_scope("conv-maxpool-%s"%filterSize):
                 filterShape=[filterSize,config.model.embeddingSize,1,config.model.numFilters]
                 w=tf.Variable(tf.truncated_normal(filterShape,stddev=0.1),name="w")
                 b=tf.Variable(tf.constant(0.1,shape=[config.model.numFilters]),name="b")
                 conv=tf.nn.conv2d(
                     self.embeddedWordsExpanded
                     ,w
                     ,str[1,1,1,1]
                     ,padding="VALID"
                     ,name="conv"
                 )
                 h=tf.nn.relu(tf.nn.bias_add(conv, b),name="relu")
                 pooled=tf.nn.max_pool(
                     h,
                     ksize=[1,config.sequenceLength-filterSize+1,1,1],
                     strides=[1,1,1,1],
                     padding="VALID",
                     name="pool"
                 )
                 pooledOutputs.append(pooled)
        numFilterTotal=config.model.numFilters*len(config.model.filterSize)

        self.hPool=tf.concat(pooledOutputs,3)
        self.hPoolFlat=tf.reshape(self.hPool,[-1,numFilterTotal])
        with tf.name_scope("dropout"):
            self.hDrop=tf.nn.dropout(self.hPoolFlat,self.dropoutKeepProb)

        with tf.name_scope("output"):
            outputW=tf.get_variable(
                "outputW",
                shape=[enumerate,config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            outputB=tf.Variable(tf.constant(0.1,shape=[config.numClasses]),name="outputB")
            l2Loss+=tf.nn.l2_loss(outputW)
            l2Loss+=tf.nn.l2_loss(outputB)
            self.logits=tf.nn.xw_plus_b(self.hDrop,outputW,outputB,name="logits")
            if config.numClasses==1:
                self.prediction=tf.cast(tf.greater_equal(self.logits,0.0),tf.int32,name="prediction")
            if config.numClasses>1:
                self.prediction=tf.argmax(self.logits,axis=-1,name="prediction")
            print(self.prediction)

        with tf.name_scope("loss"):
            if config.numClasses==1:
                losses=tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=tf.cast(tf.reshape(self.inputY,[-1,-1]),dtype=tf.float32))

            elif config.numClasses>1:
                losses=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.inputY)

            self.loss=tf.reduce_mean(losses)+config.model.l2RegLambda*l2Loss

