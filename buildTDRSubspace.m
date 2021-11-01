function [betaNeural2Behav,betaNeural2BehavOrth,projectedStates] = buildTDRSubspace(beforeLearningX,afterLearningX,beforeLearningN,afterLearningN, TDRoptions)
% This code performs the TDR analyses as in Sun, O'Shea et al, 2021.
    %% Necessary matlab package: download "gram-schmidt orthogonalization" on mathoworks and add gsog.m to the path.

    %% Input: 
    % beforeLearningX: behavioral variables before learning (a C by M matrix where C is the number of before-learning conditions and M is the number of variables)
    % afterLearningX: behavioral variables after learning (a C by M matrix where C is the number of after-learning conditions and M is the number of variables)
    % beforeLearningN: condition-averaged, centered neural activity before learning (a C by N matrix where C is the number of before-learning conditions and N is the number of neurons)
    % afterLearningN: condition-averaged, centered neural activity after learning (a C by N matrix where C is the number of after-learning conditions and N is the number of neurons)
    % TDRoptions is an indicator of whether one chooses to use before-learning data (1) or after-learning data (2), or both (3) to build the TDR subspace
    
    %% Output:
    % betaNeural2Behav is the matrix of un-orthogonalized coefficients that project neural activity to the TDR axes
    % betaNeural2BehavOrth is the matrix of orthogonalized coefficients that project neural activity to the TDR axes
    % projectedStates is the matrix of neural state coordinates on orthogonalized TDR axes
    
    %% Now build the TDR subspace
    switch TDRoptions
        case 1
            X = beforeLearningX;
            N = beforeLearningN;
        case 2
            X = afterLearningX;
            N = afterLearningN;
        case 3
            X = [beforeLearningX; afterLearningX];
            N = [beforeLearningN; afterLearningN];
    end
    % Get ready the design matrix (behavioral variables + intercept).
    behav = [X, ones(size(X,1),1)];

    % Regress neural data against the design matrix and compute the regression coefficients.
    betaBehav2Neural = behav\N;
    
    % Compute the TDR axes.
    betaNeural2Behav = pinv(betaBehav2Neural);
    betaNeural2Behav = betaNeural2Behav(:,1:size(X,2));
        
    % Orthogonalize the TDR axes before projection.
    betaNeural2BehavOrth = gsog(betaNeural2Behav);

    % Project before-learning and after-learning neural activity onto the TDR axes
    projectedStates = [beforeLearningN; afterLearningN] * betaNeural2BehavOrth;

end