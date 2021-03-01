function arcv = PSSVC( func_num, dim, seed, ~)




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
psr = [];

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
        
        [offspringc(:, 2*k-1), offspringc(:, 2*k)]=Crossover(p1, p2, gamma, ub(1), lb(1));
        
    end
    
    
    % Mutation
    parentm = pop(:, spm);
    offspringm = zeros(size(parentm));
    for k=1:nm 
        
        p=parentm(:, k);
        
        offspringm(:, k)=Mutate(p,mu,ub(1), lb(1));
        
    end
    

    
    % new offspring
    offspring = [offspringc offspringm];
    
    
    % classification for pre-selection
    pop = parent;
    fit = parentfit;
    
    ps = 0;
    for i=1:pop_size
        reference = parentfit(i);
        train = get_traing_data(arcv, knn, offspring(:, i)');
        train.label = train.y < reference;
        
        if sum(train.label) ~=0 && sum(train.label) ~= 1
            svcModel = fitcsvm(train.x, train.label, 'KernelFunction', 'RBF', 'KernelScale', 'auto');
            label = predict(svcModel, offspring(:, i)');
        else
            label = 1;
        end
        
        if label == 1
            offspringfit = fhd(offspring(:, i));
            arcv.x = [arcv.x; offspring(:, i)'];
            arcv.y = [arcv.y; offspringfit];
            
            fe = fe +1;
            ps = ps +1;
            if offspringfit < reference
                pop(:, i) = offspring(:, i);
                fit(i) = offspringfit;
            end
        end    
        
    end
    psr = [psr; ps];
    
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

function train = get_traing_data(arx, k, ind)

    dis = pdist2(ind, arx.x)';
    [~, index] = sort(dis);
    train.x = arx.x(index(1:k), :);
    train.y = arx.y(index(1:k));

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




