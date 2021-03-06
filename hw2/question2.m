
clear; close all; 
%% Input and initializations
N = 10;                       % Number of samples
mu = 0;                     % Mean - sample
Sigma = 2000;          % Covariance - sample
params = [1,1,0,2]; %params(0):a , params(1):b , params(2):c , params(3):d

% Generate N random numbers in range of -1,1
x = unifrnd(-1,1,1,N);

% Generate a gaussian additive noise below 
g = normrnd(mu,Sigma,[1,N]);

% get the final dataset with below equation
dataSet = params(1).*x.^3+params(2).*x.^2+params(3).*x+params(4)+g;

wTrue = params';
% PLot the equation now for simplicity
figure(1),
plot(dataSet),
drawnow;

% 4x4 Identitiy matrix
identityMatrix = [1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 1];
log10gammaList = linspace(-3,3,100);  % Array of gamma values
paramEstErrorL2Norm = linspace(-3,3,100); % Error in estimated params L2 norm

% Number of experiments to be performed
NumExperiments = 100;

% Parameter estimates for error normalization for each gamma
paramEstErrorMinL2Norm = ones(1,length(log10gammaList));
paramEstErrorQuartileL2Norm = ones(1,length(log10gammaList));
paramEstErrorMedianL2Norm = ones(1,length(log10gammaList));
paramEstErrorLastQL2Norm = ones(1,length(log10gammaList));

estParamError = ones(NumExperiments,length(log10gammaList)); 

%% MAP estimation :
for experiment = 1:NumExperiments
  % Generate a gaussian additive noise below 
  g = normrnd(mu,Sigma,[1,N]);
  % Generate N random numbers in range of -1,1
  x = unifrnd(-1,1,1,N);
  % get the final dataset with below equation
  dataSet = params(1).*x.^3+params(2).*x.^2+params(3).*x+params(4)+g;
  for indGamma=1:length(log10gammaList)
    sumA = [0 0 0 0;0 0 0 0;0 0 0 0;0 0 0 0];  
    sumB = [0 0 0 0];
    gamma = 10^log10gammaList(indGamma);  
    for i=1:N
      z_i = [ x(i)^3;x(i)^2;x(i);1];
      transposeZ = z_i * z_i';
      factor = Sigma^2/gamma^2;
      sumA = sumA + (transposeZ + factor.*identityMatrix);
      sumB = z_i * dataSet(i);
    end
    wMAP = inv(sumA)*sumB;
    estParamError(experiment,indGamma) = norm(wMAP - wTrue,2);
  end

end

disp(estParamError);
tempArray = ones(1,NumExperiments);

for gammaIndex=1:length(log10gammaList)
    for i=1:NumExperiments
        tempArray(i) = estParamError(i,gammaIndex);
    end
    paramEstErrorMinL2Norm(gammaIndex) = min(tempArray);
    paramEstErrorQuartileL2Norm(gammaIndex) = prctile(tempArray,25);
    paramEstErrorMedianL2Norm(gammaIndex) = prctile(tempArray,50)
    paramEstErrorLastQL2Norm(gammaIndex) = prctile(tempArray,75)
end

figure(2),
plot(10.^log10gammaList,paramEstErrorMinL2Norm),hold on,
drawnow;
figure(3),
plot(10.^log10gammaList,paramEstErrorQuartileL2Norm),hold on,
drawnow;
figure(4),
plot(10.^log10gammaList,paramEstErrorMedianL2Norm),hold on,
drawnow;
figure(5),
plot(10.^log10gammaList,paramEstErrorLastQL2Norm),hold on,
drawnow;
