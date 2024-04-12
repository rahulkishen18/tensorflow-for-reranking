import os
from google.cloud import bigquery


import threading

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
from tensorflow_io.bigquery import BigQueryClient
from tensorflow_io.bigquery import BigQueryReadSession
from tensorflow.python.framework import dtypes
import pdb

# Uncomment for eager mode
# tf.compat.v1.enable_eager_execution()
# tf.data.experimental.enable_debug_mode()

##### STEP 1 ###################
##### Setup the environment ####
################################
# This can be set with `mcli create env secret GCP_PROJECT_ID=XXX`
# You can create any arbitrary environment variable and have it automatically loaded
# into a run with `mcli create env secret` (see: https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/env.html)
GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
DATASET_ID = "mslr_fold1"
CHECKPOINT_PATH = "gs://mslr-connor/checkpoints/epoch"


## To make sure this is authed correctly, you'll want to run
# `mcli create secret gcp` (see: https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/gcp.html)
# which will use GCP JSON credentials to authenticate
client = bigquery.Client()


client = bigquery.Client(project=GCP_PROJECT_ID)
dataset = bigquery.Dataset(
    bigquery.dataset.DatasetReference(GCP_PROJECT_ID, DATASET_ID)
)
dataset.location = "us"

##### STEP 2 #####################
##### Pull data from BigQuery ####
# Note: I loaded Fold1 of the MSLR dataset into BigQuery
##################################
def read_bigquery(table_name):
    tensorflow_io_bigquery_client = BigQueryClient()
    read_session = tensorflow_io_bigquery_client.read_session(
        parent="projects/" + GCP_PROJECT_ID,
        project_id=GCP_PROJECT_ID,
        table_id=table_name,
        dataset_id=DATASET_ID,
        selected_fields={
            "features_flattened": {"output_type": dtypes.double, "mode": BigQueryClient.FieldMode.REPEATED},
            "label": {"output_type": dtypes.double, "mode": BigQueryClient.FieldMode.REPEATED},
            "query_id": {"output_type": dtypes.int64},
            "doc_id":{"output_type": dtypes.int64, "mode": BigQueryClient.FieldMode.REPEATED},
        },
        requested_streams=2,
    )

    dataset = read_session.parallel_read_rows()
    return dataset


BATCH_SIZE = 32
training_ds = read_bigquery("mslr_fold1")
training_ds = training_ds.map(lambda feature_map: {
      "_mask": tf.ones_like(feature_map["label"], dtype=tf.bool),
      **feature_map,
      "float_features": tf.reshape(feature_map["features_flattened"], [-1, 136]) # 136 is the original length of each row in the original tensor, easier to digest in BigQuery
  }
)
training_ds = training_ds.shuffle(buffer_size=1000).padded_batch(batch_size=BATCH_SIZE)
training_ds = training_ds.map(lambda feature_map: (
    feature_map, tf.where(feature_map["_mask"], feature_map.pop("label"), -1.)))


##### STEP 3 #####################
##### Setup async cloud checkpointing ####
##################################
### This will async upload the checkpoint
# Note: migrate to multiprocessing, this may not fully unblock the main loop
class AsyncModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super(AsyncModelCheckpoint, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        # Copy weights to avoid conflicts with ongoing training
        weights = self.model.get_weights()
        threading.Thread(target=self.save_weights, args=(weights, epoch)).start()

    def save_weights(self, weights, epoch):
        # Create a new model with the same architecture
        model_clone = tf.keras.models.clone_model(self.model)
        model_clone.set_weights(weights)
        
        # Save the cloned model's weights asynchronously
        model_clone.save_weights(self.filepath.format(epoch=epoch), save_format='tf')
        
    
# Create a model
checkpoint_path = CHECKPOINT_PATH + "_{epoch:02d}"
async_checkpoint_callback = AsyncModelCheckpoint(filepath=checkpoint_path)

# Compile and train
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    inputs = {"float_features": tf.keras.Input(shape=(None, 136), dtype=tf.float32)}
    norm_inputs = [tf.keras.layers.BatchNormalization()(x) for x in inputs.values()]
    x = tf.concat(norm_inputs, axis=-1)
    for layer_width in [128, 64, 32]:
        x = tf.keras.layers.Dense(units=layer_width)(x)
        x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)
    scores = tf.squeeze(tf.keras.layers.Dense(units=1)(x), axis=-1)


    model = tf.keras.Model(inputs=inputs, outputs=scores)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tfr.keras.losses.SoftmaxLoss(),
        metrics=tfr.keras.metrics.get("ndcg", topn=5, name="NDCG@5"),
    )
    ##### STEP 3 #####################
    ##### Train model ####
    ##################################
    model.fit(training_ds, epochs=3, callbacks=[async_checkpoint_callback])
