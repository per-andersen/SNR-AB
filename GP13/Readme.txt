Hi Per,

Below is a list of what I think each document contains:

fort.20           List of parameters of each galaxy in the sample, in the following format: ID	logmass(solar)	logSFR(solar)	number_of_hosted_SNe 
fort.25           List of (x	y	upper_limit	lower_limit	left_limit	right_limit) for the solid points with error bars in Fig. 7
fort.28           List of coordinates to plot solid line in Fig. 7
fort.29           List of coordinates to plot dashed line in Fig. 7
mannuccipnts.dat  List of coordinates corresponding to the points of Sullivan et. al. 2006 (squares in Fig. 7) 


fort.20 is my original data, from which everything else is derived - there must be something wrong if this could not be done.
You might want to subtract the second column from the third column in file fort.20 to obtain the specific star formation rates, if you wish to do things this way.
Note that nothing else is needed to reproduce Fig. 7 other than the information that is already present in the relevant papers. 

The remaining files contain the elements which I used to plot Fig. 7, which are essentially derived from fort.20 and other published parameters.
The error bars are purely root-n Poisson errors, if I remember correctly.

I hope this helps.

Yan
