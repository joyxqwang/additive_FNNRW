function [predFunc, alpha, beta, aa] = AFNN(X, Y, options)

% this function impletes AFNN : 
%        Additive Model with Feedforward Neural Network
%        min_{\beta} ||Y - H*\beta||_F^2 + r*||\beta||_F^2
%
% input: 
%        X = trainData n*d
%        Y = trainLable n*c
%        options = a structure which optionally contain the following
%               - N: numHidNodes
%               - Act: choice of activation fucntion
%               - computeAlpha: compute alpha if this value is 1
%               - r: hyperparameter for the regularization term
%
% output: 
%       predFunc = A function handle to estimate the function for new points
%       beta = weights of the hidden nodes, numHidNodes*c
%       alpha = weights of each SNP in the prediction of phenotypes
%
% Author:
%   Xiaoqian Wang
%

% tic;

[H, W, b, act] = calcuH(options,X);
[~, beta] = ridge_regress( H, Y, options );
options.W = W;
options.b = b;
predFunc = @(arg) predictAFNN(arg, options, @calcuH, beta);

[n, d] = size(X);
c = size(Y,2);
alpha = zeros(d,c);

if isfield(options,'computeAlpha')
    if options.computeAlpha == 1
        for j = 1 : d
            tmpw = X(:,j) * W(:,j)' + repmat(b',n,1);
%             tt{j} = act(tmpw);
            tmps = act(tmpw)*beta;
            aa(j,:) = sqrt(sum(tmps.*tmps));
            alpha(j,:) = sqrt(sum(tmps.*tmps)) ./ norm(X(:,j));
        end
    end
end

%

% disp(['AFNN Computation took ' num2str(toc) ' seconds.']);

end


% Predictions for AFNN
function preds = predictAFNN(X, options, kernelFunc, beta)
  H = kernelFunc(options,X);
  preds = H * beta;
end