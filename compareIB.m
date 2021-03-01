clear;
dim = 10;
knn = [1*dim, 2*dim, 3*dim, 4*dim, 5*dim];
rsm = [0.1 0.2 0.5 0.7 0.9];

run = 1;


seed = run*100 + 2019;
evolve = NoS(func, dim, seed);
filename = sprintf('nsresult/ns_result_run%d_f%d_d%d.mat', run, func, dim);
save(filename, 'evolve');


seed = run*100 + 2019;
fprintf('IBRBF is starting....')
for func = [1 2 4 8 13 15]
    fprintf('func: %d', func);
    for i = 1:5  %iterate for knn
        for j=1:5 % iterate for rsm
            evolve(i, j) = IBRBF( func, dim, seed, knn(i), rsm(j));
        end
    end
    filename = sprintf('ibresult/rbf_result_run%d_f%d_d%d.mat', run, func, dim);
    save(filename, 'evolve')
end
fprintf('\n\n\n')


fprintf('IBKNN is starting....')
seed = run*100 + 2019;
for func = [1 2 4 8 13 15]
    fprintf('func: %d', func);
    for i = 1:5  %iterate for knn
        for j=1:5 % iterate for rsm
            evolve(i, j) = IBKNN( func, dim, seed, knn(i), rsm(j));
        end
    end
    filename = sprintf('ibresult/knn_result_run%d_f%d_d%d.mat', run, func, dim);
    save(filename, 'evolve')
end
fprintf('\n\n\n')


fprintf('IBrankSVM is starting....')
seed = run*100 + 2019;
for func = [1 2 4 8 13 15]
    fprintf('func: %d', func);
    for i = 1:5  %iterate for knn
            evolve(i) = IBrankSVM( func, dim, seed, knn(i), rsm(j));
    end
    filename = sprintf('ibresult/rank_result_run%d_f%d_d%d.mat', run, func, dim);
    save(filename, 'evolve')
end



    