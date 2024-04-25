import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
import keras_nlp

BATCH_SIZE = 64
MIN_STRING_LEN = 512
SEQ_LEN = 128

EMBED_DIM = 256
FEED_FORWARD_DIM = 128
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 5000

EPOCHS = 5

# Download and extract dataset
dataset_url = "https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip"
keras.utils.get_file("simplebooks.zip", dataset_url, extract=True)
data_dir = os.path.join(os.path.expanduser("~"), ".keras", "datasets", "simplebooks")

# Load training and validation datasets
def preprocess_line(line):
    return tf.strings.length(line) > MIN_STRING_LEN

raw_train_ds = (
    tf_data.TextLineDataset(os.path.join(data_dir, "simplebooks-92-raw", "train.txt"))
    .filter(preprocess_line)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

raw_val_ds = (
    tf_data.TextLineDataset(os.path.join(data_dir, "simplebooks-92-raw", "valid.txt"))
    .filter(preprocess_line)
    .batch(BATCH_SIZE)
)

# Tokenization and preprocessing
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary_size=VOCAB_SIZE, lowercase=True)
start_packer = keras_nlp.layers.StartEndPacker(sequence_length=SEQ_LEN, start_value=tokenizer.token_to_id("[BOS]"))

def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels

train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(tf_data.AUTOTUNE)
val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(tf_data.AUTOTUNE)

# Model definition
inputs = keras.layers.Input(shape=(None,), dtype="int32")
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE, sequence_length=SEQ_LEN, embedding_dim=EMBED_DIM, mask_zero=True
)
x = embedding_layer(inputs)

for _ in range(NUM_LAYERS):
    decoder_layer = keras_nlp.layers.TransformerDecoder(num_heads=NUM_HEADS, intermediate_dim=FEED_FORWARD_DIM)
    x = decoder_layer(x)

outputs = keras.layers.Dense(VOCAB_SIZE)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])

# Train the model
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Save the model
model.save('path/to/save/model')

# Inference with text generation
def next(prompt, cache, index):
    logits = model(prompt)[:, index - 1, :]
    hidden_states = None  # You may use this later if needed
    return logits, hidden_states, cache

sampler = keras_nlp.samplers.GreedySampler()
output_tokens = sampler(next=next, prompt=start_packer(tokenizer([""])), index=1)
generated_text = tokenizer.detokenize(output_tokens)
print(f"Generated text: \n{generated_text}\n")
