#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import time

import model, sample, encoder

def interact_model(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=180,
    temperature=1, # changing the temperature changes the probabilities
    top_k=50257, # To get the probability of all possible outcomes for the next token
    top_p=1,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=40 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)


    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        logits = sample.get_logits(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        input_iter = 0
        start_time = time.perf_counter()
        while True:
            raw_text = input("Context prompt >>> ")
            word1 = input("Enter phrase 1 >>> ")
            word2 = input("Enter phrase 2 >>> ")
            enc_word1 = enc.encode(word1)
            enc_word2 = enc.encode(word2)

            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Context prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                original_context = context
                out_logits = sess.run(logits, feed_dict={
                    context: [context_tokens for _ in range(batch_size)] #Context is a placeholder that contains the encoded versions of the input text
                })
                for i in range(batch_size):
                    generated += 1
                    total_chance = 0
                    for token in enc_word1: #Summing up probabilities for phrase 1
                        #This finds the logit value for the next token and add it to the current probability
                        prob = out_logits.item(token)
                        context_tokens = np.append(context_tokens, token)
                        total_chance += prob
                        print("Probability for this token: " + str(prob) + " - totaling to: " + str(total_chance))
                        out_logits = sess.run(logits, feed_dict={
                            context: [context_tokens for _ in range(batch_size)] #Context_tokens will continually be added upon as we iterate through the phrase
                        })

                    print("For phrase 1 - encoded as: " + str(enc_word1))
                    print("Logarithmic probability is: " + str(total_chance))
                    print("Average is: " + str(total_chance/len(enc_word1)))
                    print("=" * 80)
                
                    context = original_context
                    out_logits = sess.run(logits, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })
                    total_chance = 0
                    for token in enc_word2: #Summing up probabilities for phrase 2
                        #Find the logit value for the next token and add it to the current probability
                        prob = out_logits.item(token)
                        context_tokens = np.append(context_tokens, token)
                        total_chance += prob
                        print("Probability for this token: " + str(prob) + " - totaling to: " + str(total_chance))
                        out_logits = sess.run(logits, feed_dict={
                            context: [context_tokens for _ in range(batch_size)] #Context is a placeholder that contains the encoded versions of the input text
                        })
                    print("For phrase 2 - encoded as: " + str(enc_word2))
                    print("Logarithmic probability is: " + str(total_chance))
                    print("Average is: " + str(total_chance/len(enc_word2)))

            time_elapsed = time.perf_counter()-start_time
            print('\n' + str(time_elapsed) + " seconds elapsed" + '\n' + '-' * 60 + '\n')
            start_time = time.perf_counter()
            print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)

#Sum up the logits for all the tokens that are put in
#Sum because they are in log space