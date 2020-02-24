% Expected risk minimization with 2 classes
clear all, close all,

% dimension of the vector under test
n = 2; 

% number of iid samples for the data distribution
N = 1000; 

mu(:,1) = [-2;0]; mu(:,2) = [2;0];

% Initialize and assume the variances to be identity matrices
Sigma(:,:,1) = [1 -0.9;-0.9 2]; Sigma(:,:,2) = [2 0.9;0.9 2];

% class priors for labels 0 and 1 respectively
p = [0.9,0.1]; 
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

figure(2), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 

% Data structures to store the true and false positive
% probability rates
truePositiveRateArray = [];
falsePositiveRateArray = [];

% Initialize the Minimum Probability of error to be 1 as below
minPerror = 1;

% Threshold where the probability would be minimum as 1
BestThreshold = 1;

% Mark the co-ordinates on ROC curve for lowest minimum probability of
% error
rocMarkerTruePositive = 0;
rocMarkerfalsePositive = 0;

discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));
  
for iterator = 1:1:10000
    
  gamma = iterator;
  
  decision = (discriminantScore >= log(gamma) - log(p(1) / p(2)));

  % probability of true negative
  ind00 = find(decision==0 & label==0); 
  p00 = length(ind00)/Nc(1); 
  
  % probability of false positive
  ind10 = find(decision==1 & label==0); 
  p10 = length(ind10)/Nc(1); 
  falsePositiveRateArray = [p10;falsePositiveRateArray];
  
  % probability of false negative
  ind01 = find(decision==0 & label==1); 
  p01 = length(ind01)/Nc(2); 
 
  
  % probability of true positive
  ind11 = find(decision==1 & label==1); 
  p11 = length(ind11)/Nc(2);
  truePositiveRateArray = [p11;truePositiveRateArray];

  % Probability of error would be sum of all incorrect probabilities
  perror = p10 + p01;
  if perror < minPerror
    minPerror = perror;
    BestThreshold = iterator;
    rocMarkerTruePositive = p11;
    rocMarkerfalsePositive = p10;
  end
  
end

disp(minPerror);
disp(BestThreshold);
disp(rocMarkerTruePositive);
disp(rocMarkerfalsePositive);

decision = (discriminantScore >= 0);

ind00 = find(decision==0 & label==0); 
ind10 = find(decision==1 & label==0); 
ind01 = find(decision==0 & label==1); 
ind11 = find(decision==1 & label==1); 

figure(2), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% Make a grid to devise a contour of boundary on Dataset
hGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),1000);
vGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),1000);

[h,v] = meshgrid(hGrid,vGrid);

discriminantScoreGridValues = log(evalGaussian(0.1*[h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-0.9*log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1)));

minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);

discriminantScoreGrid = reshape(discriminantScoreGridValues,1000,1000);
figure(2), contour(hGrid,vGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'),


figure(3), clf,
plot(falsePositiveRateArray,truePositiveRateArray), hold on,
plot(rocMarkerfalsePositive,rocMarkerTruePositive,'r*'),
title('ROC curve with marking of minimum Probability of error'),
xlabel('False Positive Probability Rate'), ylabel('True Positive Probability Rate'), 

% Inline function to evaluate the gaussian PDF data for each X
function g = evalGaussian(x,mu,Sigma)
  % Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
  [n,N] = size(x);
  C = ((2*pi)^n * det(Sigma))^(-1/2);
  E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
  g = C*exp(E);
end 