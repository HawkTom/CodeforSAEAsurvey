// Rank-SVM.cpp : Defines the entry point for the console application.
//

/*
#include "stdafx.h"
#include <fstream>
#include "dvector.h"
#include <iostream>
#include "mytimer.h"
*/

#include "C:\Program\MATLAB\R2016b\extern\include\mex.h"
#include "C:\Program\MATLAB\R2016b\extern\include\matrix.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>

enum KernelType
{
	kt_RBF_Euclidean = 0,
	kt_RBF_Mahalanobis = 1
};

void OptimizeL(int ntrain, double* p_Ci, double epsilon, int niter,
						 double* p_Kij,double* p_dKij,double* p_alpha,double* p_sumAlphaDKij,double* p_div_dKij)
{
	int nAlpha = ntrain-1;
	double old_alpha, new_alpha, delta_alpha, sumAlpha, dL;
	int i,i1,j;

	for (i=0; i<nAlpha; i++)
		for (j=0; j<nAlpha; j++)
			p_dKij[i*nAlpha + j] = p_Kij[i*ntrain + j] - p_Kij[i*ntrain + (j+1)] - p_Kij[(i+1)*ntrain + j] + p_Kij[(i+1)*ntrain + (j+1)];

	for (i=0; i<nAlpha;i++)
	{
	//	p_alpha[i] = p_Ci[i];// * rand()/(float)RAND_MAX;//p_Ci[i] * (0.95 + 0.05*rand()/(float)RAND_MAX);	// p_Ci[i] * rand()/(float)RAND_MAX;
	//	p_alpha[i] = p_Ci[i] * rand()/(float)RAND_MAX;
		p_alpha[i] = p_Ci[i] * (0.95 + 0.05*rand()/(float)RAND_MAX);
	}

	for (i=0; i<nAlpha; i++)
	{
		sumAlpha = 0;
		for (j=0; j<nAlpha;j++)
			sumAlpha += p_alpha[j] * p_dKij[i*nAlpha + j];
		p_sumAlphaDKij[i] = -(sumAlpha - epsilon) / p_dKij[i*nAlpha + i];
	}

	for (i=0; i<nAlpha; i++)
		for (j=0; j<nAlpha; j++)
			p_div_dKij[i*nAlpha + j] = p_dKij[i*nAlpha + j] / p_dKij[j*nAlpha + j];

	for (i=0; i<niter; i++)
	{	
		i1 = i%nAlpha;	//	int i1 = rand()%nAlpha;
		old_alpha = p_alpha[i1];
		new_alpha = old_alpha + p_sumAlphaDKij[i1];
		if (new_alpha > p_Ci[i1])		new_alpha = p_Ci[i1];
		if (new_alpha < 0)				new_alpha = 0;
		delta_alpha = new_alpha - old_alpha;

		dL = delta_alpha * p_dKij[i1*nAlpha + i1] * ( p_sumAlphaDKij[i1] - 0.5*delta_alpha + epsilon);

		if (dL > 0)
		{
			for (j=0; j<nAlpha; j++)
				p_sumAlphaDKij[j] -= delta_alpha * p_div_dKij[i1*nAlpha + j];

			p_alpha[i1] = new_alpha;
		}
	}
}

double DistPow2_Euclidean(double* x1, double* x2,int nx)
{
	double tmp;
	double dist = 0;
	for (int i=0; i<nx; i++)
	{
		tmp = x1[i] - x2[i];
		dist += tmp * tmp;
	}
	return dist;
}

double DistPow2_Mahalanobis(double* x1, double* x2, double* tmpdx, double* Cinv,int nx)
{
	double dx1_new;
	int ps;
	double dist = 0;
	for (int i=0; i<nx; i++)
		tmpdx[i] = x1[i] - x2[i];

	for (int i=0; i<nx; i++)
	{
		dx1_new = 0;
		ps = i*nx;
		for (int j=0; j<nx; j++)
			dx1_new += tmpdx[j] * Cinv[ps + j];
		dist += dx1_new * tmpdx[i];
	}
	return dist;
}


void Encoding(double* x, double* invsqrtC, double* xmean, int npoints, int nx)	//x'(i) = C^(-0.5) * ( x(i) - xmean(i) )
{
	double* xcur;
	double sum;
	int jrow;
	double* dx = new double[nx];
	for (int i=0; i<npoints; i++)
	{
		xcur = &x[i*nx];
		for (int j=0; j<nx; j++)
			dx[j] = xcur[j] - xmean[j];
		for (int j=0; j<nx; j++)
		{
			sum = 0;
			jrow = j*nx;
			for (int k=0; k<nx; k++)
				sum += invsqrtC[jrow + k] * dx[k];
			xcur[j] = sum;
		}
	}
	delete[] dx;
}

void Learning(double* x_training, double* y_training_svm, double* x_test, double* y_test_svm, 
		int nx, int ntrain, int ntest, int niter, KernelType kernel, double epsilon, double* p_Ci, double* p_Cinv,
		double sigma_A, double sigma_Pow, double* xmean, int doEncoding)

{
	//0. Init Temp Data
	int nAlpha = ntrain-1;
	double* p_Kij = new double[ntrain*ntrain];		double* p_dKij = new double[nAlpha*nAlpha];
	double* p_alpha = new double[nAlpha];			double* p_sumAlphaDKij = new double[nAlpha];
	double* p_div_dKij = new double[nAlpha*nAlpha];	double* p_dx = new double[nx];
	double* Kvals = new double[ntrain];

	double ttotal = 0;

	//TIMER_START(1);
	//1.We can transform our points to the new coordinate system and 
	//then calculate Euclidean distance instead of 'EXPENSIVE' Mahalanobis distance
	if (doEncoding == 1)
	{	//let's suppose that we know that we want to do encoding so input invC is the C^-0.5 ;)
		Encoding(x_training, p_Cinv, xmean, ntrain, nx); 
		Encoding(x_test, p_Cinv, xmean, ntest, nx);
	}
	//cout << "encoding:\t\t";	ttotal += TIMER_FINISH(1);
	

	//TIMER_START(1);
	//2. Calculate the distance between points, then calculate sigma(gamma) and Kernel Matrix
	double distPow2;
	double avrdist = 0;
	for (int i=0; i<ntrain; i++)
	 for (int j=i; j<ntrain; j++)
		if (i == j)		p_Kij[i*ntrain + j] = 0;
		else
		{
			if (kernel == kt_RBF_Euclidean)				distPow2 = DistPow2_Euclidean(&x_training[i*nx], &x_training[j*nx], nx);
			else if (kernel == kt_RBF_Mahalanobis)		distPow2 = DistPow2_Mahalanobis(&x_training[i*nx], &x_training[j*nx], p_dx, p_Cinv, nx);
			if (distPow2 < 0)	printf("distPow2 < 0:%f\n",distPow2);
			avrdist += sqrt(distPow2);
			p_Kij[i*ntrain + j] = p_Kij[j*ntrain + i] = distPow2;//dist;
		}
	avrdist = avrdist / ((ntrain-1)*ntrain/2);		//average distance
	double sigma = sigma_A * pow(avrdist , sigma_Pow);
	double TwoSigmaPow2 = 2.0*sigma*sigma;
	for (int i=0; i<ntrain; i++)
		for (int j=i; j<ntrain; j++)
		{
			if (i == j)		p_Kij[i*ntrain + j] = 1.0;
			else			p_Kij[i*ntrain + j] = p_Kij[j*ntrain + i] = exp(- p_Kij[i*ntrain + j] / TwoSigmaPow2);
		}
	//cout << "kernel:\t\t";	ttotal += TIMER_FINISH(1);


	//TIMER_START(1);
	//3. Optimize alpha parameters
	OptimizeL(ntrain, p_Ci, epsilon, niter, p_Kij, p_dKij, p_alpha, p_sumAlphaDKij, p_div_dKij);
	//cout << "optimization :\t";
	//ttotal += TIMER_FINISH(1);
		

	//TIMER_START(1);
	//4.1 Calculate Fsvm for training points using the Kernel matrix which we already know
	for (int i=0; i<ntrain; i++)
	{
		double Fit = 0;
		for (int j=0; j<ntrain-1; j++)
			if (p_alpha[j] != 0)
				Fit += p_alpha[j] *( p_Kij[i*ntrain + j] - p_Kij[i*ntrain + j+1] ); // Kvals[j] - Kvals[j+1], while Kvals[j] = p_Kij[i*ntrain + j];
		y_training_svm[i] = Fit;
	}

	//4.2 Calculate Fsvm for test points, it is very expensive for Mahalanobis distance because it's (x1-x2)*invC*(x1-x2)'
	for (int i=0; i<ntest; i++)
	{
		for (int j=0; j<ntrain-1; j++)
		{
			bool calculate = true;	//only support vectors
			if (j == 0)											{	if (p_alpha[j] == 0) calculate = false;		}
			else if ((p_alpha[j] == 0) &&  (p_alpha[j-1] == 0))		calculate = false;
			if (calculate)
			{
				if (kernel == kt_RBF_Euclidean)
						distPow2 = DistPow2_Euclidean(&x_test[i*nx], &x_training[j*nx], nx);
				else if (kernel == kt_RBF_Mahalanobis)
						distPow2 = DistPow2_Mahalanobis(&x_test[i*nx], &x_training[j*nx], p_dx, p_Cinv, nx);
				Kvals[j] = exp(- distPow2 / TwoSigmaPow2);
			}
		}

		double Fit = 0;
		for (int j=0; j<ntrain-1; j++)
			if (p_alpha[j] != 0)
				Fit += p_alpha[j] *( Kvals[j] - Kvals[j+1] );
		y_test_svm[i] = Fit;
	}
	//cout << "test :\t\t";	ttotal += TIMER_FINISH(1);
	//cout << "total : \t\t" << " time:" << ttotal << '\n';

	delete[] p_sumAlphaDKij;	delete[] p_div_dKij;	delete[] p_dKij;	delete[] p_Kij;
	delete[] p_dx;				delete[] p_alpha;		delete[] Kvals;		
}



//[y_training, y_test] = RankSVM(x_training', x_test', nx, ntraining, ntest, ...
//							   niter, epsilon, Ci, kernel, Cinv, sigmaA, sigmaPow, xmean, doEncoding, verbose);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	//input
	int verbose = (int)(mxGetScalar(prhs[14]));

	double* X_training = mxGetPr(prhs[0]);
	double* X_test = mxGetPr(prhs[1]);
	int nx = (int)(mxGetScalar(prhs[2]));					if (verbose == 1) printf("nx = %d\n",nx);
	int ntraining = (int)(mxGetScalar(prhs[3]));			if (verbose == 1) printf("ntraining = %d\n",ntraining);
	int ntest = (int)(mxGetScalar(prhs[4]));				if (verbose == 1) printf("ntest = %d\n",ntest);
	int niter = (int)(mxGetScalar(prhs[5]));				if (verbose == 1) printf("niter = %d\n",niter);
	double epsilon = (double)(mxGetScalar(prhs[6]));		if (verbose == 1) printf("epsilon = %f\n",epsilon);
	double* p_Ci = mxGetPr(prhs[7]);
	int kernel = (int)(mxGetScalar(prhs[8]));				if (verbose == 1) printf("kernel = %d\n",kernel);
	double* p_Cinv = mxGetPr(prhs[9]);
	double sigma_A = (double)(mxGetScalar(prhs[10]));		if (verbose == 1) printf("sigma_A = %f\n",sigma_A);
	double sigma_Pow = (double)(mxGetScalar(prhs[11]));		if (verbose == 1) printf("sigma_Pow = %f\n",sigma_Pow);
	double* xmean = mxGetPr(prhs[12]);
	int doEncoding = (int)(mxGetScalar(prhs[13]));			if (verbose == 1) printf("doEncoding = %d\n",doEncoding);

	//
	int rowLen = mxGetN(prhs[0]);							if (verbose == 1) printf("rowLen=%d\n",rowLen);
	int colLen = mxGetM(prhs[0]);							if (verbose == 1) printf("colLen=%d\n",colLen);
	if ((ntraining != rowLen)||(nx != colLen)) 
	{	 // MatLab uses [i*colLen + j] notation, while we use [i*rowLen + j], so .. :) 
		printf("Small error: the matrix 'x_training' should have 'nx' rows and 'ntraining' columns");
		return;
	}

	//output
	plhs[0] = mxCreateDoubleMatrix(ntraining, 1, mxREAL);
	double* Y_training = mxGetPr(plhs[0]);

	plhs[1] = mxCreateDoubleMatrix(ntest, 1, mxREAL);
	double* Y_test = mxGetPr(plhs[1]);


	Learning(X_training, Y_training, X_test, Y_test, nx, ntraining, ntest, niter, (KernelType)kernel, epsilon, 
						p_Ci, p_Cinv, sigma_A, sigma_Pow, xmean, doEncoding);


}
