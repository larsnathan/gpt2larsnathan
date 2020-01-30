#!/usr/bin/env python3

from interactive_conditional_samples import interact_model
import fire


def interactive_test():

    starting_temp = 40 #0.4 * 100
    ending_temp = 60 #0.6 * 100
    temp_incr = 10 #0.1 & 100
    for i in range(starting_temp, ending_temp, temp_incr):
        temp = i / 100.0
        interact_model(temperature=temp)
    print("Finished")

        
if __name__ == '__main__':
    interactive_test()