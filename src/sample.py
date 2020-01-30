import tensorflow as tf

import model
import sys

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
        print(lm_output)
        logits = lm_output['logits'][:, :, :hparams.n_vocab] #The logits from the model
        presents = lm_output['present'] #The present from the model
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        #Past is Tensor("sample_sequence/while/Identity_1:0", shape=(1, 12, 2, 12, ?, 64), dtype=float32)
        #Prev is Tensor("sample_sequence/while/Identity_2:0", shape=(1, ?), dtype=int32)
        #Output is Tensor("sample_sequence/while/Identity_3:0", shape=(1, ?), dtype=int32)
        #Logits is Tensor("sample_sequence/while/Select:0", shape=(1, 50257), dtype=float32) - Is this the weights?
        #Samples is Tensor("sample_sequence/while/multinomial/Multinomial:0", shape=(1, x), dtype=int32) - Where x is num_samples
        #Keep experimenting with variables here
        def body(past, prev, output):
            next_outputs = step(hparams, prev, past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p) #The logits are the actual weights as far as I'm aware.
            # vvv This part is what we need to change to get the weights instead of samples
            samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32) #I think num_samples changes the number of results we get
            print(samples)
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.concat([output,[[0]], samples], axis=1) #This is tokens from the while loop.
                #Basically, output already has all the existing samples, and we are recursively adding on to the array of samples.
            ]

        past, prev, output = body(None, context, context)

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens
