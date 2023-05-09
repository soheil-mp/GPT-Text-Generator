
# Import the libraries
import re, os, string, random
import numpy as np
import tensorflow as tf

# Causal attention mask function
def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the future tokens to ensure that the attention mechanism only uses 
    information from current and previous tokens.
    """

    # Initialize the indices for i and j
    i, j = tf.range(n_dest)[:, None], tf.range(n_src)

    # Mask
    mask = i >= (j - n_src + n_dest)

    # Expand the mask
    mask = tf.cast(mask, dtype)

    # Reshape the mask
    mask = tf.reshape(mask, [1, n_dest, n_src])

    # Only keep the upper half of the dot prodict matrix
    matrix_mult = tf.concat([tf.expand_dims(batch_size), tf.constant([1, 1], dtype=tf.int32)], axis=0)

    # Tile the mask
    out = tf.tile(mask, matrix_mult)

    return out

# Transforme block class
class TransformerBlock(tf.keras.layers.Layer):

    # Constructor function
    def __init__(self, embed_dim, num_heads, ff_dims, rate):

        # Inherite the parent's constructor
        super(TransformerBlock, self).__init__()

        # Self-attention layer
        self.att = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)

        # FFN layer
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dims, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])

        # Layer normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layer
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    # Call function
    def call(self, inputs, training, ):

        # Initialize th variables
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)

        # First attention block
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)

        # Second attention block
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layer_norm2(out1 + ffn_output)

        return out
    
# Token and positional embedding class
class TokenAndPositionEmbedding(tf.keras.layers.Layer):

    # Initialize the constructor function
    def __init__(self, maxlen, vocab_size, embed_dim):

        # Inherite the parent's constructorr
        super(TokenAndPositionEmbedding, self).__init__()

        # Token embedding
        self.token_embed= tf.keras.layers.Emebdding(input_dim=vocab_size, output_dim=embed_dim)

        # Position embedding layer
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    # Call function
    def call(self, x):

        # Maximum sequence length
        maxlen = tf.shape(x)[-1]

        # Initialize the positions
        positions = tf.range(start=0, limit=maxlen, delta=1)

        # Feed the positions to the position embedding layer
        positions = self.pos_emb(positions)

        # Feed the tokens tot he token emebedding layer
        x = self.token_embed(x)

        # Add the token and position embedding
        out = x + positions

        return out
    
# Initialize the hyperparameters
vocab_size = 20000
maxlen = 80
embed_dim = 256
num_heads = 2
feed_forward_dim = 256

# GPT model
def create_model():

    # Input layer 
    inputs = tf.keras.layers.Input(shape=(maxlen, ), dtype=tf.int32)

    # Token and position embedding layer
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    # Transformer blck
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)

    # Output layer
    outputs = tf.keras.layers.Dense(vocab_size)(x)

    # Construct the model
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, x])

    # Loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compile the model
    model.compile("adam", loss=[loss_fn, None], )

    return model 
