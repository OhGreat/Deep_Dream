import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-img", action="store", dest="img", type=str)
    parser.add_argument("-os", action="store", dest="octave_steps", default=0, type=int)
    parser.add_argument("-ss", action="store", dest="step_size", default=0.01, type=float)
    # extra arguments for experimentation
    parser.add_argument("-b", action="store", dest="bands", 
                        help="Can be used to set the number of bands",
                        type=int)
    parser.add_argument("-p", action="store", dest="permutations", 
                        help="Can be used to set the signature size",
                         type=int)
    parser.add_argument("-t", action="store", dest="threshold", 
                        help="Can be used to set the threshold",
                        type=float)
    parser.add_argument("-v", action="store", dest="verbose", 
                        help="Can be set to 1 to see debug messages on terminal",
                        default=0, type=int)
    args = parser.parse_args()
    print("arguments passed:",args)