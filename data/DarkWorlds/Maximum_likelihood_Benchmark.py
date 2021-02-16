""" Maximum_likelihood_Benchmark.py

This small code only works on the test set. It creates a halo at a given position x0,y0 and calculates what the ellipticity of the galaxies would look like given a 1/r model. From this it creayes likelihood for a halo given x0,y0. It then iterates over the field grid point at a time. The result is a likelihood map of nbinxnbin size grid. The position of the halo is the grid point with the maximum likelihood.

Note this code can only calculate the position of 1 halo and cannot cope with more than one halo

@Author: David Harvey
Created: 22 August 2012
"""

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

import numpy as np
import csv as c

if __name__ == "__main__":
	
	n_skies=file_len('Test_haloCounts.csv')-1 #The number of skies in total
	halos=3 #Number of halos in the most complicated sky
	position_halo=np.zeros([n_skies,2,halos],float) #The array in which
							#I will record the position
							#of each halo

	nhalo = np.zeros([n_skies],float)
	col=np.zeros([1],int)+1
	nhalo=np.loadtxt('Test_haloCounts.csv',\
			 usecols=(1,),skiprows=1,delimiter=',')

            #Read in num_halos

	for k in xrange(n_skies):
		p=k+1
		x,y,e1,e2=np.loadtxt('Test_Skies/Test_Sky%i.csv' % p,\
				     delimiter=',',unpack=True,usecols=(1,2,3,4),skiprows=1)
				#Read in the x,y,e1 and e2
				#positions of each galaxy in the list for sky number k:

		#Grid the sky up. Here I set the parameters of the grid.
		nbin=10 #Number of bins in my grid
		image_size=4200.0 #Overall size of my image
		binwidth=image_size/float(nbin) # The resulting width of each grid section
		likelihood=np.zeros([nbin,nbin],float) #The array in which I am going
						       #to store the likelihood that
						       #a halo could be at that
						       #grid point in the sky.

		for i in xrange(nbin): # I iterate over each x0
			for j in xrange(nbin): #and y0 points of the grid
            
				x0=i*binwidth #I set the proposed x position of the halo
				y0=j*binwidth #I set the proposed y position of the halo
            
				r_from_halo=np.sqrt((x-x0)**2+(y-y0)**2)
				# I find the distance each galaxy is from
				#the proposed halo position
				angle_wrt_centre=np.arctan((y-y0)/(x-x0))
				#I find the angle each galaxy is at with
				#respects to the centre of the halo.               
				force=1./r_from_halo
				#Assuming that the force dark matter has
				#on galaxies is 1/r i find the distortive
				#force the dark matter has on each galaxy
        
				e1_force=-1.*force*np.cos(2.*angle_wrt_centre)
				# work out what this force is in terms of e1
				e2_force=-1.*force*np.sin(2.*angle_wrt_centre) # and e2
            
				chisq=np.sum(((e1_force-e1)**2+(e2_force-e2)**2))
				# I then compare this hypotehtical e1 and e2
				#to the actual data in the sky and find the chisquare fit
				likelihood[i,j]=np.exp(-(chisq/2.))
				# I then find the likelihood that a halo is at position x0,y0
            
            
		position_halo[k,0,0] = np.where(likelihood == \
						np.max(likelihood))[0][0]*binwidth
						#I find the maximum likelihood point
						#and its associated index
		position_halo[k,1,0] = np.where(likelihood == \
						np.max(likelihood))[1][0]*binwidth
						#and then find which x and y position
						#this is in the sky.
	
	
	c = c.writer(open("Maximum_likelihood_Benchmark.csv", "wb")) #Now write the array to a csv file
	c.writerow([str('SkyId'),str('pred_x1'),str( 'pred_y1'),str( 'pred_x2'),str( 'pred_y2'),str( 'pred_x3'),str(' pred_y3')])
	for k in xrange(n_skies):
		halostr=['Sky'+str(k+1)] #Create a string that will write
					 #to the file and give the first element the sky_id
		for n in xrange(3):
			halostr.append(position_halo[k,0,n]) #Assign each of the halo x
							     #and y positions to the string
			halostr.append(position_halo[k,1,n])
		c.writerow(halostr) #Write the string to a csv file with
				    #the sky_id and the estimated positions

