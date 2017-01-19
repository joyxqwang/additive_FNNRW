function outLambda = tuneR( X, Y, numPartsKFoldCV, lambdaRange, func, options )
% this function impletes tuneR: 
%        tuning hyperparameter r via cross validation
%
% input: 
%        X = data matrix n*d
%        Y = label matrix n*c
%        numPartsKFoldCV = number of folds in cross validation
%        lambdaRange = range of hyperparameter r
%        func = choice of optimization model involing r
%        options = a structure which optionally contain settings for func
%
% output: 
%        outLambda = best choice of r w.r.t. RMSE
%
% Author:
%   Xiaoqian Wang
%

n = size(X,1);
validErr = zeros(length(lambdaRange),1);
        
for cvIter = 1:numPartsKFoldCV
	testStartIdx = round( (cvIter-1)*n/numPartsKFoldCV + 1);
    testEndIdx = round( cvIter*n/numPartsKFoldCV );
	trainIdxs = [1:(testStartIdx-1), (testEndIdx+1):n];
	testIdxs = [testStartIdx:testEndIdx]';
        
	Ytr = Y(trainIdxs, :);
	Yte = Y(testIdxs, :);
	Xtr = X(trainIdxs, :);
    Xte = X(testIdxs, :);
    
    for ii = 1 : length(lambdaRange)
        options.r = lambdaRange(ii);
        predFunc = func(Xtr, Ytr, options);
        YPred = predFunc(Xte);
        validErr(ii) = validErr(ii) + ...
            sqrt(norm(YPred-Yte,'fro')^2/(norm(Yte,'fro')^2));
%         validErr(ii) = validErr(ii) + ...
%             sqrt(norm(YPred-Yte,'fro')^2/(norm(Yte,'fro')^2)/numel(Yte));
    end
    
end
    
[~,idx] = min(validErr);
outLambda = lambdaRange(idx);

end

