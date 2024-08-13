import os
import tensorflow as tf

print("TF_ENABLE_ONEDNN_OPTS:", os.getenv('TF_ENABLE_ONEDNN_OPTS'))
print(tf.config.list_physical_devices('GPU'))
