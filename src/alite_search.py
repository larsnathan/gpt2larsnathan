#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import time

import model, sample, encoder

# How it should function:
# It is similar to the beam search,
# but it only keeps track of the edges of the tree, instead of keeping track of every context
# You take the beam_width highest probability contexts that are currently in the array of available contexts,
# then replace it with the beam_width top k appended to it


def alite_search(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=10,
    temperature=1, # .5 usually has numbered steps, .7 usually does not
    beam_width=3,
    max_contexts=100,
    top_k=None,
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

        # o_file = open("outbeam.txt", "a")
        # o_file.write(model_name + " - Beam Width " + str(beam_width) + '\n')

        input_iter = 0
        while True:
            raw_text = ""
            if (input_iter < len(input_samples)):
                raw_text = input_samples[input_iter]
                input_iter += 1
                print(raw_text)
            elif (len(input_samples) == 0):
                raw_text = input("Model prompt >>> ")

            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            start_time = time.perf_counter()
            generated = 0
            times_run = 0
            for _ in range(nsamples // batch_size):
                
                for i in range(batch_size):
                    generated += 1
                    max_length = len(context_tokens) + length
                    contexts = [context_tokens]
                    # o_file.write(str(contexts) + '\n')
                    print(contexts)
                    probability_map = {}
                    
                    while True: #This will probably check if the context lengths are less than max_length
                        new_contexts = []

                        #Get highest probability context
                        max_key = str(contexts[0])
                        if (bool(probability_map)):
                            max_key = max(probability_map.keys(), key=(lambda k: probability_map[k]))
                            print("Max key: " + max_key + " --- with probability: " + str(probability_map[max_key]))
                        else:
                            probability_map[str(context_tokens)] = 0

                        if (len(probability_map) >= max_contexts):
                            break
                        
                        #Find the highest probability context
                        con = max_key
                        str_values = con.strip('][').split(', ')
                        new_values = []
                        for val in str_values:
                            new_values.append(int(val))
                        con = new_values
                        
                    
                        out_logits = sess.run(logits, feed_dict={
                            context: [con for _ in range(batch_size)]
                        })

                        times_run += 1
                        
                        # Normalize the outputs
                        out_logits = out_logits - np.max(out_logits)
                        eo_logits = np.exp(out_logits) + 1e-20
                        out_logits = np.log(  eo_logits / (np.sum(eo_logits))   )

                        logit_indeces = []
                        logit_probs = []
                        for logit_index in range(len(out_logits[0])):
                            if (out_logits.item(logit_index) > np.min(out_logits)):
                                #We should get (beam width) # of logit indeces and probabilities
                                logit_indeces.append(logit_index)
                                logit_probs.append(out_logits[0].item(logit_index))

                        for i in range(len(logit_indeces)):
                            temp_context = con.copy()
                            temp_context.append(logit_indeces[i])
                            if str(con) in probability_map.keys():
                                probability_map[str(temp_context)] = probability_map[str(con)] + logit_probs[i]
                                # o_file.write("Probability for " + str(temp_context) + " is " + str(logit_probs[i]) + " + " + str(probability_map[str(con)]) + " = " + str(probability_map[str(temp_context)]) + "\n")
                                # print("Probability for " + str(temp_context) + " is " + str(logit_probs[i]) + " + " + str(probability_map[str(con)]) + " = " + str(probability_map[str(temp_context)]))

                            else:
                                probability_map[str(temp_context)] = logit_probs[i]
                                # o_file.write("Probability for " + str(temp_context) + " is " + str(logit_probs[i]) + " = " + str(probability_map[str(temp_context)]) + "\n")
                                # print("Probability for " + str(temp_context) + " is " + str(logit_probs[i]) + " = " + str(probability_map[str(temp_context)]))

                            new_contexts.append(temp_context)

                        if str(con) in probability_map.keys():
                            del probability_map[str(con)]

                        contexts = new_contexts
                        new_probs = {}
                        for con in contexts:
                            if str(con) in probability_map:
                                new_probs[str(con)] = probability_map[str(con)]
                                                        
                        # for con in contexts:
                        #     print(enc.decode(con) + " --- Probability: ---" + str(probability_map[str(con)]))

                        string_contexts = list(probability_map.keys())
                        #print(string_contexts)
                        new_contexts = []
                        for con in string_contexts:
                            str_values = con.strip('][').split(', ')
                            new_values = []
                            for val in str_values:
                                new_values.append(int(val))
                            new_contexts.append(new_values)

                        contexts = new_contexts 
                        
                    
                    sorted_probs = dict(sorted(probability_map.items(), key=lambda x: x[1], reverse=True))
                    string_contexts = list(sorted_probs)
                    new_contexts = []
                    for con in string_contexts:
                        str_values = con.strip('][').split(', ')
                        new_values = []
                        for val in str_values:
                            new_values.append(int(val))
                        new_contexts.append(new_values)


                    for context in new_contexts:
                        con_string = enc.decode(context)
                        # o_file.write(con_string + '\n')
                        print(con_string + " -- Length: " + str(len(context)) + " -- Probability: " + str(probability_map[str(context)]))

                    # print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + '\n')
                    
                    # print(text)
                    
                    time_elapsed = time.perf_counter()-start_time
                    # o_file.write('\n' + str(time_elapsed) + " seconds elapsed" + '\n' + '-' * 60 + '\n')
                    print('\n' + str(time_elapsed) + " seconds elapsed" + '\n' + '-' * 60 + '\n')
                    print(str(times_run) + " iterations run")
                    print (str(len(new_contexts)) + " contexts outputted")
                    return

            # o_file.write("=" * 80 + '\n')
            print("=" * 80)

    


if __name__ == '__main__':
    fire.Fire(alite_search)

#Some observations: We get logits (num_contexts/(beam_width-1)) times