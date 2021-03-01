function [ftrain_Svm, ftest_Svm] = GetSVMRank_FAST(x_training, x_test, ntraining, ntest, nx, doEncoding ,niter)
epsilon = 1;
nAlpha = ntraining - 1;
Cvalue = 1e+6;
Ci = zeros(nAlpha,1);
z = 1:nAlpha;
Ci(z) = Cvalue*((nAlpha-z).^2.0 );
sigmaA = 1.0;          
sigmaPow = 1.0;
if (doEncoding == 0)    % 0 - without Encoding          
    invC = eye(nx);
    kernel = 0;         %0 - Euclidian RBF
end;
if (doEncoding == 1)    % 0 - without Encoding  
    invC = eye(nx);
    kernel = 0;         %0 - Euclidian RBF
end;
verbose = 0;            % 0 - without comments
xmean = zeros(nx,1);
[ftrain_Svm0, ftest_Svm0] = RankSVM(x_training', x_test', nx, ntraining, ntest, ...
       niter, epsilon, Ci, kernel, invC, sigmaA, sigmaPow, xmean, doEncoding, verbose);
ftrain_Svm=ftrain_Svm0;
ftest_Svm =ftest_Svm0;
end

