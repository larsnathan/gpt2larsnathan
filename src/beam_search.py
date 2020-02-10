#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import time

import model, sample, encoder

# How it should function:
# We give the model a context, same as with the interactive samples program
# For each loop, we take the $top_k highest probability results
# We then take each of these possibilities and append them to the context to make $top_k unique contexts
# Keep doing this until we have $length number of tokens appended to the context
# The complexity should be $top_k ^ $length

#Ideas:
# - Make it more efficient by going a certain length (a variable like $partition_size), and then collapsing it down to the most likely sentence, and then continuing
# - Should the algorithm be completely run with the Tensor object in sample.py or converted to ndarray in this file?
# - The temperature will probably have to be fairly low
# - Should we use length and top_k variables or make seperate ones?
# - Could be recursive: for k in top_k: context += k; next_tokens(k, context);

# def recursive_search(tokens, current_prob, max_length, max_prob, sess, logits):
#     context = tf.placeholder(tf.int32, [1, None])
#     if (len(tokens) >= max_length):
#         #See if it is the highest probability and replace the context_tokens if so
#         if (current_prob > max_prob):
#             max_prob = current_prob
#             max_tokens = tokens
#         return

#     out_logits = sess.run(logits, feed_dict={
#                 context: [tokens for _ in range(1)]
#             })

#     for logit_index in range(out_logits.size):
#         if (out_logits.item(logit_index) > -100000000000.000):
#             new_context = np.append(tokens, logit_index)
#             current_prob += out_logits.item(logit_index)
#             recursive_search(new_context, current_prob, max_length, max_prob, sess, logits)

def beam_search(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=30,
    temperature=1, # .5 usually has numbered steps, .7 usually does not
    beam_width=3,
    top_k=3,
    top_p=1,
    models_dir='models',
    input_samples=[],
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

    top_k = beam_width #Set the top_k to the beam_width to only find the beam_width number of logits

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
            raw_text = ""
            if (input_iter < len(input_samples)):
                raw_text = input_samples[input_iter]
                input_iter += 1
                print(raw_text)
            elif (len(input_samples) == 0):
                raw_text = input("Model prompt >>> ")

            time_elapsed = time.perf_counter()-start_time

            print('\n' + str(time_elapsed) + " seconds elapsed" + '\n' + '-' * 60 + '\n')

            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                
                for i in range(batch_size):
                    generated += 1
                    max_length = len(context_tokens) + length
                    contexts = [context_tokens]
                    print(contexts)
                    probability_map = {}
                    
                    while True: #This will probably check if the context lengths are less than max_length
                        new_contexts = []
                        if (len(contexts[0]) == max_length):
                            break
                        for con in contexts:
                            out_logits = sess.run(logits, feed_dict={
                                context: [con for _ in range(batch_size)]
                            })

                            logit_indeces = []
                            logit_probs = []
                            for logit_index in range(len(out_logits[0])):
                                if (out_logits.item(logit_index) > -10000000000.000):
                                    #We should get (beam width) # of logit indeces and probabilities
                                    logit_indeces.append(logit_index)
                                    logit_probs.append(out_logits[0].item(logit_index))


                            for i in range(len(logit_indeces)):
                                temp_context = con.copy()
                                temp_context.append(logit_indeces[i])
                                # print(con)
                                # print(temp_context)
                                if str(con) in probability_map:
                                    probability_map[str(temp_context)] = probability_map[str(con)] + logit_probs[i]
                                else:
                                    probability_map[str(temp_context)] = logit_probs[i]
                                new_contexts.append(temp_context)
                        
                        contexts = new_contexts
                        new_probs = {}
                        for con in contexts:
                            if str(con) in probability_map:
                                new_probs[str(con)] = probability_map[str(con)]
                        top_probs = dict(sorted(new_probs.items(), key=lambda x: x[1], reverse=True)[:beam_width]) #Gets the top beam_width probabilities off the top
                        string_contexts = list(top_probs.keys())
                        new_contexts = []
                        for con in string_contexts:
                            str_values = con.strip('][').split(', ')
                            new_values = []
                            for val in str_values:
                                new_values.append(int(val))
                            new_contexts.append(new_values)
                        contexts = new_contexts
                                                    
                        
                    print(contexts)
                    
                    for context in contexts:
                        con_string = enc.decode(context)
                        print(con_string)

                    # print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + '\n')
                    
                    # print(text)
                    return

            print("=" * 80)

    
        
    


if __name__ == '__main__':
    fire.Fire(beam_search)

#For every step, keep the same width (ex. 3)
#Just keep track of a set of contexts that is beam width long