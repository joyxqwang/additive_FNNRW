# additive_FNNRW
Code for the additive model via feedforward neural networks with random weights.
This function optimizes the following problem:

    min_{\beta} ||Y - H_X*\beta||_F^2 + r*||\beta||_F^2

Format of input:

    n: number of samples
    d: number of SNPs
    c: number of QTs
    X: n*dim SNP data
    Y: n*c phenotype matrix
    options: a structure which optionally contain the following
        - N: numHidNodes
        - Act: choice of activation fucntion
        - computeAlpha: compute alpha if this value is 1
        - r: hyperparameter for the regularization term
                
Format of output:

    predFunc: A function handle to estimate the function for new points
    beta: weights of the hidden nodes, numHidNodes*c
    alpha: weights of each SNP in the prediction of phenotypes

Simply run the code in matlab as below:

    [predFunc, alpha] = AFNN(X, Y, options);
