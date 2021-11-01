% This code performs the TDR analyses and defines the uniform shift axis as in Sun, O'Shea et al, 2021.

%% Neural and behavioral variables needed for running this script:
% beforeLearningX: behavioral variables before learning (a C by M matrix where C is the number of before-learning conditions and M is the number of variables)
% afterLearningX: behavioral variables after learning (a C by M matrix where C is the number of after-learning conditions and M is the number of variables)
% beforeLearningN: condition-averaged neural activity before learning (a C by N matrix where C is the number of before-learning conditions and N is the number of neurons)
% afterLearningN: condition-averaged neural activity before learning (a C by N matrix where C is the number of before-learning conditions and N is the number of neurons)

%% Center the neural data and build TDR subspaces.
beforeLearningN_center = beforeLearningN - mean([beforeLearningN;afterLearningN], 1);
afterLearningN_center = afterLearningN - mean([beforeLearningN;afterLearningN], 1);
TDRoption = 1;
[betaNeural2Behav,betaNeural2BehavOrth,projectedStates] = buildTDRSubspace(beforeLearningX,afterLearningX,beforeLearningN_center,afterLearningN_center,TDRoption);

% Plot neural states in a TDR subspace (for example, the 2D force-predictive TDR subspace)
n_cond = size(projectedStates,1) / 2; % Assume there're equal numbers of conditions before and after learning.
colormap = load('colormap.mat');
figure; 
for i=1:n_cond
    plot(projectedStates(i,1),projectedStates(i,2),'.','MarkerSize',30,'Color',colormap(i,:)); 
    hold on; 
    plot(projectedStates(i+n_cond,1),projectedStates(i+n_cond,2),'d','MarkerSize',10,'MarkerFaceColor',colormap(i,:)); 
    hold on;
end

%% Define the uniform-shift learning axis 
% Connecting before-learning and after-learning centroids
uniformAxis = mean(afterLearningN) - mean(beforeLearningN);
uniformAxis = uniformAxis';

% Assume we built a 2D force-predctive TDR subspace, now orthogonalize the uniform shift against the TDR axes.
uniformAxisOrth = uniformAxis - betaNeural2BehavOrth(:,1) * dot(betaNeural2BehavOrth(:,1), uniformAxis)/norm(betaNeural2BehavOrth(:,1))^2;
uniformAxisOrth = uniformAxisOrth - betaNeural2BehavOrth(:,2) * dot(betaNeural2BehavOrth(:,2), uniformAxisOrth)/norm(betaNeural2BehavOrth(:,2))^2; 

% Normalize the uniform shift axis.
uniformAxisOrthNorm = uniformAxisOrth / norm(uniformAxisOrth);

% Project neural activity onto the uniform-shift axis.
uniformStates = [beforeLearningN_center; afterLearningN_center] * uniformAxisOrthNorm;