#!/usr/bin/env python3

from probability_prediction import probability_model
import fire


def probability_test():

    models = ["124M", "355M", "1558M"]
    context = "To bake a cake, the first step is"
    phrases = [" to take the vacuum from the closet", "ef we nks if s se"]

    for model in models:
        starting_temp = 40 #0.4 * 100
        ending_temp = 150 #1.4 * 100
        temp_incr = 20 #0.1 & 100
        for i in range(starting_temp, ending_temp, temp_incr):
            temp = i / 100.0
            probability_model(temperature=temp, model_name=model, input_context=context, input_phrases=phrases)
        print("Finished for model: " + model + "\n\n")

        
if __name__ == '__main__':
    probability_test()