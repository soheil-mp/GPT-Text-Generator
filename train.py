import tensorflow as tf
from tensorflow import keras

def causal_attention_mask(batch_size, dest_seq_len, src_seq_len, dtype):
    i, j = tf.range(dest_seq_len)[:, None], tf.range(src_seq_len)
    mask = i >= (j - src_seq_len + dest_seq_len)
    mask = tf.cast(mask, dtype)
    mask = tf.reshape(mask, [1, dest_seq_len, src_seq_len])
    matrix_mult = tf.concat([tf.expand_dims(batch_size), tf.constant([1, 1], dtype=tf.int32)], axis=0)
    out = tf.tile(mask, matrix_mult)
    return out

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.feed_forward_network = tf.keras.Sequential([
            tf.keras.layers.Dense(feed_forward_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.multi_head_attention(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layer_norm1(inputs + attention_output)
        ffn_output = self.feed_forward_network(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layer_norm2(out1 + ffn_output)
        return out

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_embedding = tf.keras.layers.Emebdding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=max_seq_len, output_dim=embed_dim)

    def call(self, x):
        max_seq_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_seq_len, delta=1)
        positions = self.position_embedding(positions)
        x = self.token_embedding(x)
        out = x + positions
        return out

vocab_size = 20000
max_seq_len = 80
embed_dim = 256
num_heads = 2
feed_forward_dim = 256
dropout_rate = 0.1

def create_transformer_model():
    inputs = tf.keras.layers.Input(shape=(max_seq_len, ), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(max_seq_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim, dropout_rate)
    x = transformer_block(x)
    outputs = tf.keras.layers.Dense(vocab_size)(x)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, x])
    model.compile("adam", loss=[loss_fn, None])
    return model

# Initialize the model
model = create_transformer_model()
print(model.summary())