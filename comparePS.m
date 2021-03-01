clear;
warning('off')
addpath(genpath('surrogates/'))

dim = 10;
psr = [2 3 5 7 10];
run = 1;


seed = run*100 + 2019;
fprintf('PSRBF is starting....')
for func = [1 2 4 8 13 15]
    fprintf('func: %d', func);    
    for j=1:5 % iterate for pres-selection rate
        evolve(j) = PSRBF( func, dim, seed, psr(j));
    end
    filename = sprintf('psresult/rbf_result_run%d_f%d_d%d.mat', run, func, dim);
    %     save(filename, 'evolve')
end
fprintf('\n\n\n')



seed = run*100 + 2019;
fprintf('PSKNN is starting....')
for func = [1 2 4 8 13 15]
    fprintf('func: %d', func);    
    for j=1:5 % iterate for pres-selection rate
        evolve(j) = PSKNN( func, dim, seed, psr(j));
    end
    filename = sprintf('psresult/rbf_result_run%d_f%d_d%d.mat', run, func, dim);
    %     save(filename, 'evolve')
end
fprintf('\n\n\n')


seed = run*100 + 2019;
fprintf('PSrankSVM is starting....')
for func = [1 2 4 8 13 15]
    fprintf('func: %d', func);    
    for j=1:5 % iterate for pres-selection rate
        evolve(j) = PSrankSVM( func, dim, seed, psr(j));
    end
    filename = sprintf('psresult/rank_result_run%d_f%d_d%d.mat', run, func, dim);
    %     save(filename, 'evolve')
end
fprintf('\n\n\n')


seed = run*100 + 2019;
fprintf('PSSVC is starting....')
for func = [1 2 4 8 13 15]
    fprintf('func: %d', func);    
    evolve(j) = PSSVC( func, dim, seed, 0);
    filename = sprintf('psresult/svc_result_run%d_f%d_d%d.mat', run, func, dim);
    %     save(filename, 'evolve')
end
fprintf('\n\n\n')


