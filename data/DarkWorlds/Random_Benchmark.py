""" Random Position Estimator

This code just randomly guesses the positions of halos given the number of halos

@Author: David Harvey
Created: 6th September 2012
"""
import numpy as np
import random as rd
import csv as c

def file_len(fname):
	""" Calculate the length of a file
	Arguments:
	       Filename: Name of the file wanting to count the rows of
	Returns:
	       i+1: Number of lines in file
	"""

	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1




if __name__ == "__main__":


  n_skies=file_len('Test_haloCounts.csv')-1 # Test set only, doesnt train
     
  position_halo=np.zeros([n_skies,2,3],float) #Set up the array in which I will
                                                #assign my estimated positions
    
  nhalo=np.loadtxt('Test_haloCounts.csv',\
                   usecols=(1,),delimiter=',',skiprows=1) #Load in the number
                                                          #of halos for each sky



  c = c.writer(open("Random_Benchmark.csv", "wb")) #Now write the array to a csv file
  c.writerow([str('SkyId'),str('pred_x1'),str( 'pred_y1'),str( 'pred_x2'),str( 'pred_y2'),str( 'pred_x3'),str(' pred_y3')])
  for k in xrange(n_skies):

    for n in xrange(int(nhalo[k])):
        position_halo[k,0,n]=rd.random()*4200.
        position_halo[k,1,n]=rd.random()*4200.
        
    halostr=['Sky'+str(k+1)] #Create a string that will write to the file
                             #and give the first element the sky_id
    for n in xrange(3):
      halostr.append(position_halo[k,0,n]) #Assign each of the
                                           #halo x and y positions to the string
      halostr.append(position_halo[k,1,n])
    c.writerow(halostr) #Write the string to a csv
                        #file with the sky_id and the estimated positions
    

  

