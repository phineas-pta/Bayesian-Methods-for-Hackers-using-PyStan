""" Gridded_Signal_benchmak.py

This small code grids up the sky and then finds the signal in each bin. The signal in this case is the force tangential to the galaxy which is e_tangential=-(e1cos(theta)+e2sin(theta)) where theta is the angle between the galaxy and the proposed centre of the halo. From this the pixel with the largest signal should contain the halo. In the case of more than one halo, we should find that the top two or three bins in the gridded area should contain the or 2 or 3 halos. Therefore this code should be able to find all three halos.

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

    n_skies=file_len('Test_haloCounts.csv')-1 # Test set only, doesnt train
    
    position_halo=np.zeros([n_skies,2,3],float) #Set up the array in which I will
                                                #assign my estimated positions
    
    nhalo=np.loadtxt('Test_haloCounts.csv',\
        usecols=(1,),delimiter=',',skiprows=1) #Load in the number of halos for each sky
            
    for k in xrange(n_skies):
        p=k+1
        #Read in the x,y,e1 and e2 positions of
        #each galaxy in the list for sky number k:
        x,y,e1,e2=np.loadtxt('Test_Skies/Test_Sky%i.csv'\
                             % p,delimiter=',',unpack=True,usecols=(1,2,3,4),skiprows=1)

        #So I need to grid the sky up. Here I set the parameters of the grid.
        nbin=15 #Number of bins in my grid
        image_size=4200.0 #Overall size of my image
        binwidth=float(image_size)/float(nbin) # The resulting width of each grid section
    
        average_tan_force=np.zeros([nbin,nbin],float) #Set up the signal array
                                                      #in which Im going to find
                                                      #the maximum of.
    
        for i in xrange(nbin):
            for j in xrange(nbin):
            
                x0=i*binwidth+binwidth/2. #I set the proposed x position of the halo
                y0=j*binwidth+binwidth/2. #I set the proposed y position of the halo
            
                angle_wrt_halo=np.arctan((y-y0)/(x-x0)) #I find the angle each
                                                    #galaxy is at with respects
                                                    #to the centre of the halo.               
                tangential_force=-(e1*np.cos(2.0*angle_wrt_halo)\
                               +e2*np.sin(2.0*angle_wrt_halo))
                               #Find out what the tangential force
                               #(or signal) is for each galaxy with
                               #respects to the halo centre, (x0,y0)
                tangential_force_in_bin=tangential_force[(x >= i*binwidth) & \
                                                     (x < (i+1)*binwidth) & \
                                                     (y >= j*binwidth) & \
                                                     (y < (j+1)*binwidth)]
                                    #Find out which galaxies lie within the gridded box


                if len(tangential_force_in_bin) > 0:
                    average_tan_force[i,j]=sum(tangential_force_in_bin)\
                        /len(tangential_force_in_bin) #Find the average signal per galaxy
                else:
                    average_tan_force[i,j]=0
                

        index=np.sort(average_tan_force,axis=None) #Sort the grid into the
                                                   #highest value bin first,
                                                   #which should be the centre
                                                   #of one of the halos
        index=index[::-1] #Reverse the array so the largest is first
        for n in xrange(int(nhalo[k])):
            position_halo[k,0,n]=np.where(average_tan_force\
                                          == index[n])[0][0]\
                                          *binwidth
                                          #For each halo in the sky find
                                          #the position and assign
            position_halo[k,1,n]=np.where(average_tan_force\
                                          == index[n])[1][0]\
                                          *binwidth
                                          #The three grid bins
                                          #with the highest signal should
                                          #contain the three halos.
    c = c.writer(open("Gridded_Signal_benchmark.csv", "wb")) #Now write the array to a csv file
    c.writerow([str('SkyId'),str('pred_x1'),str( 'pred_y1'),str( 'pred_x2'),str( 'pred_y2'),str( 'pred_x3'),str(' pred_y3')])
    for k in xrange(n_skies):
        halostr=['Sky'+str(k+1)] #Create a string that will write to the file
                      #and give the first element the sky_id
        for n in xrange(3):
            halostr.append(position_halo[k,0,n]) #Assign each of the
                                             #halo x and y positions to the string
            halostr.append(position_halo[k,1,n])
        c.writerow(halostr) #Write the string to a csv
                        #file with the sky_id and the estimated positions


       

