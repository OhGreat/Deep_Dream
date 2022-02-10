import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-img", action="store", dest="img", type=str)

    parser.add_argument("-ss", action="store", dest="step_size", 
                        help="Sets the step_size per iteration",
                        default=0.01, type=float)

    parser.add_argument("-or", action="store", dest="octaves_range", 
                        help="Set octave range",
                        default=(-1,4), type=tuple)

    parser.add_argument("-os", action="store", dest="steps_per_octave",
                        help="Sets the number of steps per octave",
                        default=80, type=int)
    
    parser.add_argument("-osc", action="store", dest="octave_scale", 
                        help="Set scaling for each octave step",
                        default=1.2, type=float)

    args = parser.parse_args()
    print("arguments passed:",args)



if __name__ == "__main__":
    main()