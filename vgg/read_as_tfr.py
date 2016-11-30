#Cavs lost to Bucks
#   Sucks
#30Nov2016
#
"""
Conver the image and label to tfRecord for easy processing
"""
import tensorflow as tf
import os
import utils
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# configuration
# currently i assume the ImageNet dataset consists of tons of image files and a single corresponding label file
num_total_images = 0
num_classes = 0
num_batch_size = 20
num_test_size = 500
path_dataset = "./img/"
checkpoint_dir = "./"
# path_dataset = "dataset/ImageNet/"
learning_rate = 0.001
mode = sys.argv[1]

class to_one_hot:
    """Convert list of labels to one-hot encoding"""
    def __init__(self):
        self.l_encoder = LabelEncoder()
        self.oh_encoder = OneHotEncoder()

    def fit_transform(self, label_list):
        int_encoded = self.l_encoder.fit_transform(label_list)
        int_encoded = int_encoded.reshape([-1, 1])
        oh_encoded = self.oh_encoder.fit_transform(int_encoded)
        return oh_encoded.toarray().astype(int).tolist()

    def transform(self, label_list):
        int_encoded = self.l_encoder.transform(label_list)
        int_encoded = int_encoded.reshape([-1, 1])
        oh_encoded = self.oh_encoder.transform(int_encoded)
        return oh_encoded.toarray().astype(int).tolist()
        


# load training image_path & labels
# at this stage, just load filename rather than real data
dataset_images = list()
dataset_labels = list() # for test
test_paths_labels = list()
for subdir in os.listdir(path_dataset):
    if subdir.startswith('.') or "test" == subdir:
        continue
    elif os.path.isfile(path_dataset + subdir):
        test_paths_labels.append(path_dataset + subdir)
        continue
    for image_file_name in os.listdir(path_dataset + subdir):
        if image_file_name.startswith('.'):
            continue
        image = path_dataset + subdir + '/' + image_file_name
        dataset_images.append(image)
        dataset_labels.append(subdir) # for test
        num_total_images += 1
num_classes = len(set(dataset_labels)) # for test
text_classes = list(set(dataset_labels)) # for test

encoder = to_one_hot()
dataset_labels = encoder.fit_transform(dataset_labels)

#dataset_images contains list of images path
#dataset_labels contains list of correstponding labels

writer = tf.python_io.TFRecordWriter("train.tfrecords")


for i, img_path in enumerate(dataset_images):
    print i
    img = utils.load_image(img_path)
    img = img.reshape((1, 224, 224, 3))
    img_raw = img.tobytes() 
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=dataset_labels[i])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))

    writer.write(example.SerializeToString()) 


writer.close()


#Read TFRecords
def parse_sequence_example(serialized, image_feature, label_feature):
  """Parses a tensorflow.SequenceExample into an image and caption.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.
    image_feature: Name of SequenceExample context feature containing image
      data.
    label_feature: Name of label

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    caption: A 1-D uint64 Tensor with dynamically specified length.
  """
  context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features={
          image_feature: tf.FixedLenFeature([], dtype=tf.string)
      },
      sequence_features={
          label_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })

  encoded_image = context[image_feature]
  caption = sequence[label_feature]
  return encoded_image, caption

def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.

  In training the capacity of the queue is important because a larger queue
  means better mixing of training examples between shards. The minimum number of
  values kept in the queue is values_per_shard * input_queue_capacity_factor,
  where input_queue_memory factor should be chosen to trade-off better mixing
  with memory usage.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        /tmp/train_data-?????-of-00100).
    is_training: Boolean; whether prefetching for training or eval.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  if is_training:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  tf.scalar_summary(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue

def build_input(input_file_pattern='./train.tfrecords', 
    batch_size = 20, 
    values_per_input_shard=1000, 
    input_queue_capacity_factor=2, 
    num_input_reader_threads = 2
    )
      reader = tf.TFRecordReader()
      input_queue = prefetch_input_data(
          reader,
          input_file_pattern,
          is_training=True,
          batch_size=batch_size,
          values_per_shard=values_per_input_shard,
          input_queue_capacity_factor=input_queue_capacity_factor,
          num_reader_threads=num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(num_input_reader_threads):
        serialized_sequence_example = input_queue.dequeue()
        image, caption = parse_sequence_example(
            serialized_sequence_example,
            image_feature='image',
            label_feature='label')

        images_and_captions.append([image, caption])

      # Batch inputs.
      queue_capacity = (2 * self.config.num_input_reader_threads *
                        self.config.batch_size)
      images, input_seqs, target_seqs, input_mask = (
          input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity))

    self.images = images
    self.input_seqs = input_seqs
    self.target_seqs = target_seqs
    self.input_mask = input_mask