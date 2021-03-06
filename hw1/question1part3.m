clear all, close all,

% Generate 2-dimensional data vectors from 2 Gaussian pdfs
% total number of their sample account to 10000
n = 2; 
N1 = 6000; 
factor1 = -1*ones(n,1); 
class1 = 3*(rand(n,n)-0.5); 

N2 = 4000; 
factor2 = 1*ones(n,1); 
class2 = 2*(rand(n,n)-0.5);

class1Data = class1*randn(n,N1)+factor1*ones(1,N1);
class2Data = class2*randn(n,N2)+factor2*ones(1,N2);

% Estimate mean vectors and covariance matrices from samples
meanClass1 = mean(class1Data,2); 
varianceClass1 = cov(class1Data');

meanClass2 = mean(class2Data,2); 
varianceClass2 = cov(class2Data');

% Calculate the between/within-class scatter matrices
% assuming equal weights for in and between class scatter matrices
Sb = (meanClass1-meanClass2)*(meanClass1-meanClass2)';
Sw = varianceClass1 + varianceClass2;

% Solve for the Fisher LDA projection vector (in w)
[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

figure(2), clf,
plot(class1Data(1,:),class1Data(2,:),'o'), hold on,
plot(class2Data(1,:),class2Data(2,:),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels before FDA projection'),
xlabel('x_1'), ylabel('x_2'), 

% Linearly project the data from both categories on to w
ProjectionClass1 = (w'*class1Data);
ProjectionClass2 = (w'*class2Data);

% Compute the difference of mean of class 2 and 1 . 
% Based on the delta compute the sign to flip projected matrix
% to ease the job of decision making . 
signMatrix = sign(mean(ProjectionClass2) - mean(ProjectionClass1));

% Linearly project the data from both categories on to w
ProjectionClass1 = signMatrix * ProjectionClass1;
ProjectionClass2 = signMatrix * ProjectionClass2;

figure(3), clf,
plot(ProjectionClass1(1,:),zeros(1,N1),'o'), hold on,
plot(ProjectionClass2(1,:),zeros(1,N2),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels post FDA projection'),
xlabel('x_1'), ylabel('x_2'), 

minPerror = 1;
thresholdBestPerformance = -100;
truePositivePredictionArray = [];
falsePositivePredictionArray = [];
rocMarkerTruePositivePredictionRate = 1;
rocMarkerFalsePositivePredictionRate = 1;

% Vary the threshold from -100 to 100 and apply the fischer LDA classifier
for thresholdIterator = -100:100
    decisionFischerLDAClass1 = (ProjectionClass1 <= thresholdIterator);
    decisionFischerLDAClass2 = (ProjectionClass2 >= thresholdIterator);
    
    pC1SamplesPredicted = length(find(decisionFischerLDAClass1 == 1))/N1;
    pc1error = 1 - pC1SamplesPredicted;
    
    pC2SamplesPredicted = length(find(decisionFischerLDAClass2 == 1))/N2;
    
    truePositivePredictionArray = [truePositivePredictionArray;pC2SamplesPredicted];
    falsePositivePredictionArray = [falsePositivePredictionArray;pc1error];
    
    pc2error = 1 - pC2SamplesPredicted;
    perror = pc1error + pc2error;
    
    if perror < minPerror
        minPerror = perror;
        thresholdBestPerformance = thresholdIterator;
        rocMarkerTruePositivePredictionRate = pC2SamplesPredicted;
        rocMarkerFalsePositivePredictionRate = pc1error;
    end
end


disp(minPerror);
disp(thresholdBestPerformance);
disp(rocMarkerTruePositivePredictionRate);
disp(rocMarkerFalsePositivePredictionRate);

figure(4), clf,
plot(falsePositivePredictionArray,truePositivePredictionArray), hold on,
plot(rocMarkerFalsePositivePredictionRate,rocMarkerTruePositivePredictionRate,'r*'),
title('ROC curve with marking of minimum Probability of error'),
xlabel('False Positive Probability Rate'), ylabel('True Positive Probability Rate'), 
