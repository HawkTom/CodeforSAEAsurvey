function arcv = PSrankSVM( func_num, dim, seed, psr)

rng(seed);

lb = -100 * ones(dim, 1);
ub = 100 * ones(dim, 1);

fhd = @(x)(cec15problems('eval',x,func_num));


maxfe = 200 *dim;
pop_size =40;

% LHS database initilization
arcv.x = repmat(lb', 5*dim, 1) + lhsdesign(5*dim, dim, 'iterations', 1000) .* (repmat(ub' - lb', 5*dim, 1));
arcv.y = eval_pop(fhd, arcv.x')';

[arcv.y, index] = sort(arcv.y);
arcv.x = arcv.x(index, :);
fit = arcv.y(1:pop_size)';
pop = arcv.x(1:pop_size, :)';
fe = 5*dim;

% hyperparameter for GA
pc=0.7;                                  % crossover percentage
nc=2*round(pc*pop_size/2);      % number of offsprings (also Parnets)
gamma=0.4;                          % extra range factor for crossover

pm=0.3;                                 % mutation percentage
nm=round(pm*pop_size);           % number of mutants
mu=0.3;                                  % mutation rate


% sort population
[fit, index] = sort(fit);
pop = pop(:, index);

fprintf('FE: %d, Fitness: %.2e \n', fe, min(fit))

knn = 5 * dim;  %training size for each surrogate model 

% ++++ ratio of pre-selection++++
psr;

% main loop
while fe < maxfe 
    
    ssr = randperm(pop_size);
    parent = pop(:, ssr);
    parentfit = fit(:, ssr);
    
    % select parents for crossover
    spc = ssr(1: nc);
    % select parents for mutation
    spm = ssr(nc+1: end);
    
    % Crossover
    parentc = pop(:, spc);
    offspringc = zeros(size(parentc));
    for k=1:nc/2
        
        p1=parentc(:, 2*k-1);
        p2=parentc(:, 2*k);
        
        temp = zeros(dim, 2*psr);
        for q = 1:psr
            [temp(:, 2*q-1), temp(:, 2*q)] = Crossover(p1, p2, gamma, ub(1), lb(1));
        end
        
        % Pre-selection
       
        train = rank_traing_data(arcv, knn, temp');
        [train.y, index] = sort(train.y);
        train.x = train.x(index, :);
        
        tnum = floor(size(train.x, 1) * 0.8);
        validate_x = train.x(tnum+1:end, :);
        validate_y = train.y(tnum+1:end, :);
        test_x = [temp'; validate_x];       % 20% train data are used for validation 
        train_x = train.x(1:tnum, :);          % 80% train data are used for traing       
        train_y = train.y(1:tnum, :);     
        ntraing = size(train_x, 1); ntesting = size(test_x, 1);
  
        [ftrain_svm, ftest_svm] = GetSVMRank_FAST(train_x, test_x, ntraing, ntesting, dim, 0 , floor(50000 * (dim^0.5)));
        
        temp_svm = ftest_svm(1: size(temp, 2));
        validate_svm = ftest_svm(size(temp, 2)+1: end);
        
        tau = corr(validate_svm, validate_y, 'type', 'kendall');
        if tau > 0.5
            [~, index] = sort(temp_svm, 'descend');
            
        else
            index = randperm(2*psr);
            
        end
        
        offspringc(:, 2*k-1) = temp(:, index(1));
        offspringc(:, 2*k) = temp(:, index(2));
        
    end
    
    
    % Mutation
    parentm = pop(:, spm);
    offspringm = zeros(size(parentm));
    for k=1:nm 
        
        p=parentm(:, k);
        
        temp = zeros(dim, psr);
        for i = 1:psr
            temp(:, i) = Mutate(p,mu,ub(1), lb(1));
        end
        
        % Pre-selection
        train = rank_traing_data(arcv, knn, temp');
        [train.y, index] = sort(train.y);
        train.x = train.x(index, :);
        
        tnum = floor(size(train.x, 1) * 0.8);
        validate_x = train.x(tnum+1:end, :);
        validate_y = train.y(tnum+1:end, :);
        test_x = [temp'; validate_x];       % 20% train data are used for validation 
        train_x = train.x(1:tnum, :);          % 80% train data are used for traing       
        train_y = train.y(1:tnum, :);     
        ntraing = size(train_x, 1); ntesting = size(test_x, 1);
  
        [ftrain_svm, ftest_svm] = GetSVMRank_FAST(train_x, test_x, ntraing, ntesting, dim, 0 , floor(50000 * (dim^0.5)));
        
        temp_svm = ftest_svm(1: size(temp, 2));
        validate_svm = ftest_svm(size(temp, 2)+1: end);
        
        tau = corr(validate_svm, -1 * validate_y, 'type', 'kendall');
        if tau > 0.5
            [~, index] = max(temp_svm);
            
        else
            index = randi(psr);
            
        end
        offspringm(:, k) = temp(:, index);
        
        
    end
    
    % new offspring
    offspring = [offspringc offspringm];
        
    % No Surrogate 
    offspringfit = eval_pop(fhd, offspring);
    fe = fe + size(offspring, 2);
    
    arcv.x = [arcv.x; offspring'];
    arcv.y = [arcv.y; offspringfit'];
    
    % Merge the parent and offspring
    pop = [parent offspring];
    fit = [parentfit offspringfit];
    
    % Select new parents
    [fit, index] = sort(fit);
    pop = pop(:, index);
    
    % Truncation
    pop = pop(:, 1:pop_size);
    fit = fit(:, 1:pop_size);
    
    fprintf('FE: %d, Fitness: %.2e \n', fe, min(arcv.y));
     
end

end

function fit = eval_pop(f, pop)
    
    [dim, n] = size(pop);
    fit = zeros(1, n);
    for i = 1:n
        fit(i) = f(pop(:, i));
    end

end



function train = rank_traing_data(arx, k, pop)
    
    dis = pdist2(pop, arx.x);
    pop_size = size(pop, 1);
    train.x = [];
    train.y = [];
%     train.x = zeros(k, size(pop, 2));zz
    for i = 1:k
        j = mod(i, pop_size); 
        if j ==0
            j=pop_size;
        end
        [~, index] = min(dis(j, :));
        train.x = [train.x; arx.x(index, :)];
        train.y = [train.y; arx.y(index, :)];
        dis(:, index) = inf;
    end

end

% crossover operator
function [y1, y2]=Crossover(x1,x2,gamma,VarMax,VarMin)

    alpha=unifrnd(-gamma,1+gamma,size(x1));
    
    y1=alpha.*x1+(1-alpha).*x2;
    y2=alpha.*x2+(1-alpha).*x1;
    
    y1=max(y1,VarMin);
    y1=min(y1,VarMax);
    
    y2=max(y2,VarMin);
    y2=min(y2,VarMax);

end

% mutation operator
function y=Mutate(x,mu,VarMax,VarMin)
    
    nVar=size(x,1);
    nmu=ceil(mu*nVar);
    
   % original mutation     
%     j=randsample(nVar,nmu)';
%     
%     sigma=0.1*(VarMax-VarMin);
%     
%     y=x;
%     y(j)=x(j)+sigma*randn(size(j));
%     
%     y=max(y,VarMin);
%     y=min(y,VarMax);
    
    % uniform mutation 
    r = rand(nVar, 1) >= mu;
    y = unifrnd(VarMin, VarMax, nVar, 1);
    y(r) = x(r);

end



