#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import time

import model, sample, encoder

def zigzag(seq):
  return seq[::2], seq[1::2]

def interact_model(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=180,
    temperature=1, # .5 usually has numbered steps, .7 usually does not
    top_k=40,
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
        output = sample.sample_sequence( #Basically, output isn't actually the values in the array, but is the tensor/function of the method sample.sample_sequence
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        ) #Output is a Tensor object from the while loop in sample sequence
        #Output[0] is a strided slice


        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        input_iter = 0
        o_file = open("output.txt", "a")
        o_file.write("Test with temperature: " + str(temperature) + '\n')
        print("Test with temperature: " + str(temperature) + '\n')
        start_time = time.perf_counter()
        while True:
            raw_text = ""
            if (input_iter < len(input_samples)):
                raw_text = input_samples[input_iter]
                input_iter += 1
                print(raw_text)
                o_file.write('\n' + raw_text + '\n')
            elif (len(input_samples) == 0):
                raw_text = input("Model prompt >>> ")
                word1 = input("Enter word 1 >>> ")
                word2 = input("Enter word 2 >>> ")
                enc_word1 = enc.encode(word1)
                enc_word2 = enc.encode(word2)

            elif (input_iter >= len(input_samples)):
                time_elapsed = time.perf_counter()-start_time

                o_file.write('\n' + str(time_elapsed) + " seconds elapsed" + '\n' + '-' * 60 + '\n')
                print('\n' + str(time_elapsed) + " seconds elapsed" + '\n' + '-' * 60 + '\n')
                break
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)] #Context is a placeholder that contains the encoded versions of the input text
                })[:, len(context_tokens):]
                out_logits = sess.run(logits, feed_dict={
                    context: [context_tokens for _ in range(batch_size)] #Context is a placeholder that contains the encoded versions of the input text
                })[:, len(context_tokens):]
                print(output.value_index)
                for i in range(batch_size):
                    generated += 1
                    print(out)
                    #clipped_logits = out_logits[out_logits != -10000000000.0]
                    # for index in range(50252):
                    #     if (out_logits.item(index) != -10000000000.0):
                    #         print(enc.decode([index]))
                    print(out_logits.argmax())
                    print("For word 1 - encoded as: " + str(enc_word1[0]))
                    print(out_logits.item(enc_word1[0]))
                    print("For word 2 - encoded as: " + str(enc_word2[0]))
                    print(out_logits.item(enc_word2[0]))
                    # out1, out2 = zigzag(out[i])
                    # text1 = enc.decode(out1) #out[i] is a numpy array with the encoded values for the samples. Size [len,]
                    # text2 = enc.decode(out2)
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + '\n')
                    o_file.write("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + '\n')
                    # print(text1)
                    # print(text2)
                    print(text)
                    try:
                        o_file.write(text + '\n')
                    except:
                        print("\nUnknown character encountered. Moving to next step\n")
                        o_file.write("\nUnknown character encountered. Moving to next step\n")
                        break
            print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)

