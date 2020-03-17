from bert.run_classifier import DataProcessor, InputExample, ColaProcessor, MnliProcessor, MrpcProcessor, XnliProcessor
import os
from bert import tokenization
import tensorflow as tf


class EmoProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        return ["neutral", "positive", "negative"]

    def _create_examples(self, lines, set_type):
        examples = []

        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            if set_type == "test":
                label = "neutral"
            else:
                label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor,
        "emlo": EmoProcessor
    }
