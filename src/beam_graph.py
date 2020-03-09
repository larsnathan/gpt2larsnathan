#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import time
import igraph
from igraph import *
import plotly.graph_objects as go


import model, sample, encoder

def beam_graph(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=10,
    temperature=1, # .5 usually has numbered steps, .7 usually does not
    beam_width=3,
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

            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
                    
            #Create a graph to map the contexts
            g = Graph(directed=True)
            g.add_vertex(str(context_tokens))

            for _ in range(nsamples // batch_size):
                
                for i in range(batch_size):
                    generated += 1
                    max_length = len(context_tokens) + length
                    contexts = [context_tokens]
                    print(contexts)
                    probability_map = {}
                    all_contexts = []
                    all_contexts.append(context_tokens)
                    
                    while True: #This will probably check if the context lengths are less than max_length
                        new_contexts = []
                        if (len(contexts[0]) == max_length):
                            break
                        for con in contexts:
                            out_logits = sess.run(logits, feed_dict={
                                context: [con for _ in range(batch_size)]
                            })
                            
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
                                all_contexts.append(temp_context)
                                if str(con) in probability_map:
                                    g.add_vertex(str(temp_context))
                                    parent_context = temp_context[0:len(temp_context)-1]
                                    g.add_edge(str(temp_context),str(parent_context))
                                    # print(g)
                                    probability_map[str(temp_context)] = probability_map[str(con)] + logit_probs[i]
                                    #print("Probability for " + str(temp_context) + " is " + str(logit_probs[i]) + " + " + str(probability_map[str(con)]) + " = " + str(probability_map[str(temp_context)]))

                                else:
                                    g.add_vertex(str(temp_context))
                                    parent_context = temp_context[0:len(temp_context)-1]
                                    g.add_edge(str(temp_context),str(parent_context))
                                    # print(g)
                                    probability_map[str(temp_context)] = logit_probs[i]
                                    #print("Probability for " + str(temp_context) + " is " + str(logit_probs[i]) + " = " + str(probability_map[str(temp_context)]))

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
                                                    

                    all_strings = []   
                    for context in all_contexts:
                        con_string = enc.decode(context)
                        all_strings.append(con_string)
                    
                    for context in contexts:
                        con_string = enc.decode(context)
                        print(con_string)

                    # print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + '\n')
                    
                    # print(text)

                    print(g.get_edgelist())

                    nr_vertices = g.vcount()
                    print(nr_vertices)

                    es = EdgeSeq(g)
                    E = [e.tuple for e in es]
                    lay = g.layout_auto()

                    v_label = list(map(str, range(nr_vertices)))
                    position = {k: lay[k] for k in range(nr_vertices)}
                    Y = [lay[k][1] for k in range(nr_vertices)]
                    M = max(Y)
                    L = len(position)
                    Xn = [position[k][0] for k in range(L)]
                    Yn = [2*M-position[k][1] for k in range(L)]
                    Xe = []
                    Ye = []
                    for edge in E:
                        Xe+=[position[edge[0]][0],position[edge[1]][0], None]
                        Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]
                    labels = v_label
                    print(labels)
                    print(len(position))
                    print(len(all_contexts))

                    def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
                        L=len(pos)
                        if len(text)!=L:
                            raise ValueError('The lists pos and text must have the same len')
                        annotations = []
                        for k in range(L):
                            annotations.append(
                                dict(
                                    text=str(text[k]), # or replace labels with a different list for the text within the circle
                                    x=pos[k][0], y=2*M-position[k][1],
                                    xref='x1', yref='y1',
                                    font=dict(color=font_color, size=font_size),
                                    showarrow=False)
                            )
                        return annotations

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=Xe,
                                    y=Ye,
                                    mode='lines',
                                    line=dict(color='rgb(210,210,210)', width=1),
                                    hoverinfo='none'
                                    ))
                    fig.add_trace(go.Scatter(x=Xn,
                                    y=Yn,
                                    mode='markers',
                                    name='bla',
                                    marker=dict(symbol='circle-dot',
                                                    size=18,
                                                    color='#6175c1',    #'#DB4551',
                                                    line=dict(color='rgb(50,50,50)', width=1)
                                                    ),
                                    text=all_strings,
                                    hoverinfo='text',
                                    opacity=0.8
                                    ))
                    axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
                                zeroline=False,
                                showgrid=False,
                                showticklabels=False,
                                )

                    fig.update_layout(title= 'Tree with Reingold-Tilford Layout',
                                annotations=make_annotations(position, all_strings),
                                font_size=12,
                                showlegend=False,
                                xaxis=axis,
                                yaxis=axis,
                                margin=dict(l=40, r=40, b=85, t=100),
                                hovermode='closest',
                                plot_bgcolor='rgb(248,248,248)'
                                )
                    fig.show()

                    time_elapsed = time.perf_counter()-start_time
                    print('\n' + str(time_elapsed) + " seconds elapsed" + '\n' + '-' * 60 + '\n')
                    return

            print("=" * 80)

    


if __name__ == '__main__':
    fire.Fire(beam_graph)