from bert import modeling
from bert import tokenization
from queue import Queue
from threading import Thread
import tensorflow as tf

class InputExample(object):
    def __init__(self,unique_id,text_a,text_b):
        self.unique_id=unique_id
        self.text_a=text_a
        self.text_b=text_b
class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids

class BertVector(object):
    def __init__(self, batch_size=100, max_seq_len=5, layer_indexes=[-2]):
        self.max_seq_len = max_seq_len
        self.layer_indexes = layer_indexes
        self.tokenizer=tokenization.FullTokenizer(vocab_file="./bert_model/chinese_L-12_H-768_A-12/vocab.text")
        self.batch_size=batch_size
        self.estimator=self.get_estimator()
        self.input_queue=Queue(maxsize=1)
        self.output_queue=Queue(maxsize=1)

        self.predict_thread=Thread(target=self.predic_from_queue,daemon=True)
        self.predict_thread.start()

    def get_estimator(self,bert_config):
        def model_fn(features,labels,modes,params):

            unique_ids=features["unique_ids"]
            input_ids=features["input_ids"]
            input_mask=features["input_mask"]
            input_type_ids=features["input_type_ids"]

            model=modeling.BertModel(
                config=bert_config
            )

    def predict_from_queue(self):
        prediction=self.estimator.predict(input_fn=self.queue_predict_input_fn,yield_single_examples=False)
        for i in prediction:
            self.output_queue.put(i)


    def queue_predict_input_fn(self):
        return (tf.data.Dataset.from_generator(
            self.generate_from_queue,
            output_types={
                'unique_ids':tf.int32,
                'input_ids':tf.int32,
                'input_mask':tf.int32,
                "input_type_ids":tf.int32
            },
            output_shapes={
                "unique_ids":(None,),
                "input_ids":(None,self.max_seq_len),
                "input_mask":(None,self.max_seq_len),
                "input_type_ids":(None,self.max_seq_len)
            }
        ).prefetch(10))

    def generate_from_queue(self):
        while True:
            features=list(self.convert_examples_to_features(seq_length=self.max_seq_len,tokenizer=self.tokenizer))
            yield{
                'unique_ids': [f.unique_id for f in features],
                'input_ids': [f.input_ids for f in features],
                'input_mask': [f.input_mask for f in features],
                'input_type_ids': [f.input_type_ids for f in features]
            }
    def input_fn_builder(self,features,seq_length):
        all_unique_ids=[]
        all_input_ids=[]
        all_input_mask=[]
        all_input_type_ids=[]

        for feature in features:
            all_unique_ids
            all_input_ids.append(feature.unique_id)
            all_input_mask.append(feature.input_mask)
            all_input_type_ids.append(feature.input_type_ids)

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            num_examples = len(features)

            # This is for demo purposes and does NOT scale to large data sets. We do
            # not use Dataset.from_generator() because that uses tf.py_func which is
            # not TPU compatible. The right way to load data is with TFRecordReader.
            d = tf.data.Dataset.from_tensor_slices({
                "unique_ids":
                    tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
                "input_ids":
                    tf.constant(
                        all_input_ids, shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_mask":
                    tf.constant(
                        all_input_mask,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_type_ids":
                    tf.constant(
                        all_input_type_ids,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
            })

            d = d.batch(batch_size=batch_size, drop_remainder=False)
            return d
        return input_fn

    def convert_examples_to_features(self,seq_length,tokenizer):

        features=[]
        input_masks=[]
        examples=self._to_example(self.input_queue.get())

        for (ex_index,example) in enumerate(examples):
            tokens_a=tokenizer.tokenize(example.text_a)
            if len(tokens_a)>seq_length-2:
                tokens_a=tokens_a[0:(seq_length-2)]
            tokens=[]
            input_type_ids=[]
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids=tokenizer.convert_tokens_to_ids(tokens)
            input_mask=[1]*len(input_ids)
            input_masks.append(input_mask)
            while len(input_ids)<seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids)==seq_length
            assert len(input_mask)==seq_length
            assert len(input_type_ids)==seq_length

            if ex_index < 5:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (example.unique_id))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

            yield InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids
            )



    @staticmethod
    def _to_example(sentences):
        import re
        unique_id=0
        for ss in sentences:
            line=tokenization.convert_to_unicode(ss)
            if not line:
                continue
            line=line.strip()
            text_a=None
            text_b=None
            m=re.match(r"^(.*) \|\|\| (.*)$",line)
            if m is None:
                text_a=line
            else:
                text_a=m.group(1)
                text_b=m.group(2)
            yield InputExample(unique_id=unique_id,text_a=text_a,text_b=text_b)
            unique_id+=1
