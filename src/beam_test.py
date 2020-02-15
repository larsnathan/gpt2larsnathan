#!/usr/bin/env python3

from beam_search import beam_search
import fire


def beam_test():

    models = ["124M", "1558M"]
    context = "To bake a cake, you must"

    for model in models:
        starting_beam = 1
        ending_beam = 4
        beam_incr = 1
        
        for i in range(starting_beam, ending_beam, beam_incr):
            beam_search(model_name=model, input_samples=[context], beam_width=i)
        print("Finished for model: " + model + "\n\n")

        
if __name__ == '__main__':
    beam_test()