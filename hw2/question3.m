i = 0;
resultLikelihood1000 = [0];
resultLikelihood100 = [0];
resultLikelihood10 = [0];
while 1
  i=i+1;
  % Generate order 4 Gaussian Mixture models for 1000 samples
  for M = 1:6
    resultLikelihood1000 = [resultLikelihood1000,EMforGMM(1000,M)];
  end
  
  
  EMforGMM(10,4);
  % Generate order 4 Gaussian Mixture models for 100 samples
  for M = 1:6
    resultLikelihood10 = [resultLikelihood10,EMforGMM(10,M)];
  end
  
  EMforGMM(100,4);
    % Generate order 4 Gaussian Mixture models for 100 samples
  for M = 1:6
    resultLikelihood100 = [resultLikelihood100,EMforGMM(100,M)];
  end
  
  if i > 100
      break;
  
  end
end

x=-10000:10000;
plot(x,resultLikelihood100);
plot(x,resultLikelihood1000);
plot(x,resultLikelihood10);

  
function [result] = EMforGMM(N,M)
% Generates N samples from a specified GMM,
% then uses EM algorithm to estimate the parameters
% of a GMM that has the same nu,mber of components
% as the true GMM that generates the samples.

close all,
delta = 1e-1; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates

% Generate samples from a 4-component GMM
[alpha_true,mu_true,Sigma_true] = getParams(4);

x = randGMM(N,alpha_true,mu_true,Sigma_true);
Mbak = M;
[d,M] = size(mu_true); % determine dimensionality of samples and number of GMM components

M = Mbak;

% Bootstrap the data with 70%:30% ratio
bootstrap_ratio = 0.7;
dataValidation = x(:,bootstrap_ratio*N:N);
x = x(:,1:bootstrap_ratio*N);

N = bootstrap_ratio*N;

% Initialize the GMM to randomly selected samples
alpha = ones(1,M)/M;
shuffledIndices = randperm(N);
mu = x(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
[~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean
for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
    Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
end
t = 0; 
%displayProgress(t,x,alpha,mu,Sigma);
%pause(0.1);

[init_alpha_true,init_mu_true,init_Sigma_true] = getParams(M); 

Converged = 0; % Not converged at the beginning
while ~Converged
    for l = 1:M
        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    
    if t==1
       init_alpha_true = alphaNew;
       init_mu_true = muNew;
       init_Sigma_true = SigmaNew; 
    end
    
    plgivenx = temp./sum(temp,1);
    alphaNew = mean(plgivenx,2);
    w = plgivenx./repmat(sum(plgivenx,2),1,N);
    muNew = x*w';
    
    for l = 1:M
        v = x-repmat(muNew(:,l),1,N);
        u = repmat(w(l,:),d,1).*v;
        SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
    end
    
    Dalpha = sum(abs(alphaNew-alpha'));
    Dmu = sum(sum(abs(muNew-mu)));
    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
    alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
    t = t+1;
    %displayProgress(t,x,alpha,mu,Sigma);
    
    if t % 300 == 0
       alpha = init_alpha_true
       mu = init_mu_true
       Sigma = init_Sigma_true 
    end 
    
    if t > 100
       break;
    end
end
sumM = 0;
for m=1:M
    sumM = sumM + alpha(m)*evalGaussian(dataValidation,mu(m),Sigma(m));
end
result = [sumM];
end
%keyboard,

function [alpha,mu,variance] = getParams(M)
  if M == 1
     alpha = [1];
     mu = [-10;0];
     variance(:,:,1) = [3 1;1 20];
  elseif M == 2
     alpha = [0.7,0.3];
     mu = [-10 0 ;0 0 ];
     variance(:,:,1) = [3 1;1 20];
     variance(:,:,2) = [7 1;1 2];
  elseif M == 3
     alpha = [0.2,0.3,0.5];
     mu = [-10 0 10 ;0 0 10];
     variance(:,:,1) = [3 1;1 20];
     variance(:,:,2) = [7 1;1 2];
     variance(:,:,3) = [4 1;1 16];
  elseif M == 4
     alpha = [0.1,0.2,0.3,0.4];
     mu = [-10 0 10 2 ;0 0 10 3];
     variance(:,:,1) = [3 1;1 20];
     variance(:,:,2) = [7 1;1 2];
     variance(:,:,3) = [4 1;1 16];
     variance(:,:,4) = [5 1;1 10];
  elseif M == 5
     alpha = [0.1,0.2,0.3,0.25,0.15];
     mu = [-10 0 10 2 -50;0 0 10 3 -45];
     variance(:,:,1) = [3 1;1 20];
     variance(:,:,2) = [7 1;1 2];
     variance(:,:,3) = [4 1;1 16];
     variance(:,:,4) = [5 1;1 10];
     variance(:,:,5) = [6 1;1 14];
  else
     alpha = [0.1,0.2,0.3,0.1,0.15,0.15];
     mu = [-10 0 10 2 -50 -25;0 0 10 3 -45 20];
     variance(:,:,1) = [3 1;1 20];
     variance(:,:,2) = [7 1;1 2];
     variance(:,:,3) = [4 1;1 16];
     variance(:,:,4) = [5 1;1 10];
     variance(:,:,5) = [6 1;1 14];
     variance(:,:,6) = [8 1;1 9];
  end 
end

%%%
function displayProgress(t,x,alpha,mu,Sigma)
figure(1),
if size(x,1)==2
    subplot(1,2,1), cla,
    plot(x(1,:),x(2,:),'b.'); 
    xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
    axis equal, hold on;
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
    contour(x1Grid,x2Grid,zGMM); axis equal, 
    subplot(1,2,2), 
end
logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));
plot(t,logLikelihood,'b.'); hold on,
xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
drawnow; pause(0.1),
end

%%%
function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end

%%%
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

%%%
function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
zGMM = reshape(GMM,91,101);
%figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
end

%%%
function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end