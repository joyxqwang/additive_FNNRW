function [H, W, b, act] = calcuH(options,X)

% calcuH Computes H matrix for feedforward neural network 
% of a specified activation function on a given set of points X.
% 
% input: 
%       X = n x d data matrix 
%       (n examples, each of which is a d-dimensional vector) 
%       
%       options = a data structure with the following fields 
%       - options.Act = choice of activation function from the following:
%           'sigmoid' | 'ReLU' | 'tanh' (Default) | 'sin' | 'Gaussian'
%       - options.N = number of hidden nodes in FNN
%           Default value: 50
%       - options.k = number of interacting features
%           Default value: 1
%
% output: 
%       H = n * N matrix
%       W = weight matrix in FNN
%       b = bias in FNN
%
% Author: 
% Xiaoqian Wang 
%

%% Initialization
%
[n, d] = size(X);

%
if isfield(options,'Act')
    activation = options.Act;
else
    activation = 'tanh';
end

if isfield(options,'N')
    N = options.N;
else
    N = 50;
end

if isfield(options,'k')
    k = options.k;
else
    k = 1;
end

if isfield(options,'W')
    W = options.W;
else
    W = unifrnd(0, 1, N, d);
end

if isfield(options,'b')
    b = options.b;
else
    b = unifrnd(0, 1, N, 1);
end

%
H = zeros(n, N);


%% Calculation
%
switch activation
    
    case 'linear'
        % fprintf('sigmoid function');
        act = @(x) x;
        
    case 'sigmoid'
        % fprintf('sigmoid function');
        act = @(x) 1./(1+exp(-x));
        
    case 'ReLU'
        % fprintf('ReLU function');
        act = @(x) max(x,0);
        
    case 'tanh'
        % fprintf('tanh function');
        act = @(x) tanh(x);
        
    case 'sin'
        % fprintf('sin function');
        act = @(x) sin(x);
        
    case 'Gaussian'
        % fprintf('Gaussian function');
        act = @(x) exp(-x.^2);

    otherwise
	error('Unknown Activation Function.');
end

%
if k > 0
    tmpc = nchoosek(1:d, k);
    D = size(tmpc,1);
    tmps = zeros(n, D);
    for t = 1 : N
        tmpw = W(t,:);
        for m = 1 : k
            tmpid = tmpc(:,m);
            tmps = tmps + X(:,tmpid).*repmat(tmpw(tmpid),n,1) + ...
                repmat(b(t),n,D);
        end
        H(:,t) = sum(act(tmps),2);
    end
else
    H = act(X*W'+repmat(b',n,1));
end

end
