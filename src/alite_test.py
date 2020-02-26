#!/usr/bin/env python3

from alite_search import alite_search
import fire


def alite_test():

    models = ["124M"]#, "1558M"]
    context = "The first step to baking cookies is"
    #max_contexts = 40
    beam_width = 6

    for model in models:
        starting_max = 40
        ending_max = 71
        max_incr = 10
        
        for i in range(starting_max, ending_max, max_incr):
            alite_search(model_name=model, input_samples=[context], beam_width=beam_width, max_contexts=i)
        print("Finished for model: " + model + "\n\n")

        
if __name__ == '__main__':
    alite_test()