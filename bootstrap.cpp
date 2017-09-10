#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <time.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>


std::vector<double> read_to_vector(std::string filename)
{/*Read single column data file to vector*/
    double read_val;
    std::vector<double> Result;
    
    std::ifstream myfile(filename.c_str());
    if (!myfile)
    {
      std::cout << "!!!Could not read file: " << filename << std::endl;
      exit(-1);
    }
    while (myfile.good())
    {
        myfile >> read_val;
        Result.push_back(read_val);
    }
    myfile.close();
    return Result;
}

std::vector<std::vector<double> > read_to_2d_vector(std::string filename, int n_rows, int n_cols)
{/*Read two column data file to vector of vectors*/
    
    
    double read_val;
    std::vector< std::vector<double> > Result(n_rows);
    
    //Resizing output vector
    for (int i=0; i<n_rows; i++)
    {
        Result[i].resize(n_cols);
    }
    
    //Checking that file exists and can be read, if able then file is open
    std::ifstream myfile(filename.c_str());
    if (!myfile)
    {
      std::cout << "!!!Could not read file: " << filename << std::endl;
      exit(-1);
    }
    
    //Pushing data into vectors
    for (int i=0; i<n_rows; i++)
    {
        for (int j=0; j<n_cols; j++)
        {
            myfile >> read_val;
            Result[i][j] = read_val;
        }
    }
    
    //Closing file
    myfile.close();
    
    return Result;
}

std::vector<double> linspace(double a, double b, int n)
{
    std::vector<double> array;
    double step = (b-a) / (n-1);

    while(a <= b) {
        array.push_back(a);
        a += step;           // could recode to better handle rounding errors
    }
    return array;
}

std::vector<double> nicelog_snr(std::vector<double> ssfr, double a, double k, double ssfr0, double alpha)
{

	std::vector<double> snr(ssfr.size());

	for(int ii=0; ii<ssfr.size(); ii++)
	{
		snr[ii] = a + a * log10(ssfr[ii]/ssfr0 + alpha) / k;
	}
	return snr;
}

double lnlike(double a, double k, double ssfr0, double alpha, std::vector<double> ssfr, std::vector<double> snr, std::vector<double> snr_err)
{
	double sum = 0.;

	std::vector<double> snr_model = nicelog_snr(ssfr,a,k,ssfr0,alpha);

	for(int ii=0; ii<ssfr.size(); ii++)
	{
		sum = sum + pow((snr[ii] - snr_model[ii]) / snr_err[ii],2);
	}
	return 0.;
}


std::vector<double> run_grid()
{
	int resolution = 20;
	double a_min = 5.0e-15; double a_max = 7.0e-13;
	double k_min = 0.01; double k_max = 3.;
	double s0_min = 1.0e-11; double s0_max = 15.0e-10;
	double alpha_min = 0.01; double alpha_max = 1.5;

	std::vector<double> a_par = linspace(a_min,a_max,resolution);
	std::vector<double> k_par = linspace(k_min,k_max,resolution);
	std::vector<double> s0_par = linspace(s0_min,s0_max,resolution);
	std::vector<double> alpha_par = linspace(alpha_min,alpha_max,resolution);

}


int main()
{
 //Setting t0
 clock_t t_start = clock();

 //Setting up PRNG
 const gsl_rng_type * T;
 gsl_rng * r;
 gsl_rng_env_setup();
 T = gsl_rng_default;
 r = gsl_rng_alloc (T);

// Setting up data
 std::string ROOT_DIR = "/Users/perandersen/Data/";
 std::vector<std::vector<double> > smith = read_to_2d_vector("Mathew/Smith_2012_Figure5_Results.txt", 6, 4);
 std::vector<std::vector<double> > sullivan = read_to_2d_vector("Mathew/Smith_2012_Figure5_Sullivan_Results.txt", 6, 3);

 std::vector<double> logssfr;
 std::vector<double> snr;
 std::vector<double> snr_err;

 for (int ii=0; ii<smith.size(); ii++)
 {
 	logssfr.push_back(smith[ii][0]);
 	snr.push_back(smith[ii][1]);
 	snr_err.push_back(pow( pow(smith[ii][2],2) + pow(smith[ii][3],2) ,0.5 ));
 	std::cout << "Value: " << logssfr[ii] << std::endl;
 }

 for (int ii=0; ii<sullivan.size(); ii++)
 {
 	logssfr.push_back(sullivan[ii][0]);
 	snr.push_back(sullivan[ii][1]);
 	snr_err.push_back(sullivan[ii][2]);
 	std::cout << "Value: " << logssfr[ii+smith.size()] << std::endl;
 }

 //std::cout << "Value: " << sullivan[5][0] << std::endl;

}