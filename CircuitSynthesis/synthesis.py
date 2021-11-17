from squigglylines import * 

import random
import numpy as np

def bridgecircuit(componentsize):
    pass
def starcircuit(componentsize):
    pass
def deltacircuit(componentsize):
    pass
def main():
    mu = 20 #mü is the amount of average parts
    sigma = 5 #sigma is standart deviation

    partcount = int(np.random.normal(mu, sigma, 1))
    if partcount < 3:
        partcount = 3

    componentSize = int(random.randint(64,256))
    # randomly select if it should be a Component Line or SubCombonent

    
    
    

if __name__ == '__main__':
    main()