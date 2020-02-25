
clear; close all; 
%% Input and initializations
N = 10;                       % Number of samples
mu = [1;0];                     % Mean - sample
Sigma = [1,0.3;0.3,1];          % Covariance - sample
SigmaV = 2*0.5;                 % V8ariance - 0-mean Gaussian noise
nRealizations = 100;            % Number of realizations for the ensemble analysis
 
gammaArray = 10.^[-10:0.1:5];   % Array of gamma values

T = [0.4 -0.3;-0.3 0.8];         % Coefficients for cubic terms
A = [0.4 -0.3;-0.3 0.8];        % Coefficients for quadratic terms
b = [0; 0];                     % Coefficients for linear terms 
c = -2;                         % Constant

%% Generate 2D Gaussian data, compute value of the function
% Draw N samples of x from a Gaussian distribution
x = mvnrnd(mu,Sigma,N)';

% Calculate y: quadratic in x + additive 0-mean Gaussian noise
y = yFunc(x,T,A,b,c) + SigmaV^0.5*randn(1,N);

%% Visualize the surface and data
[x1Grid,x2Grid] = meshgrid(linspace(min(x(1,:)),max(x(1,:)),100),...
    linspace(min(x(2,:)),max(x(2,:)),100));
xGrid = [x1Grid(:),x2Grid(:)]';
yGrid = reshape(yFunc(xGrid,T,A,b,c),size(x1Grid));

figure('units','normalized','outerposition',[0.01 0.04 0.99 0.95]); 
hold on; box on; grid on;
s=surf(x1Grid,x2Grid,yGrid); s.LineStyle = 'none';
xlabel('x1'); ylabel('x2'); zlabel('y');
view(3); colormap(parula); colorbar;
lgnd=legend(['True surface: ']);
lgnd.Location = 'northeast';

% Define z vectors for linear and quadratic models
zL = [ones(1,size(x,2)); x(1,:); x(2,:)];
zQ = [ones(1,size(x,2)); x(1,:); x(2,:); x(1,:).^2; x(1,:).*x(2,:); x(2,:).^2];

% Compute z*z^T for linear and quadratic models
for i = 1:N
    zzTL(:,:,i) = zL(:,i)*zL(:,i)';
    zzTQ(:,:,i) = zQ(:,i)*zQ(:,i)';
end

%% MAP Parameter estimation: \theta \sim \mathcal{N}(0,\gamma \mathrm{I})
for i = 1:length(gammaArray)
    gamma = gammaArray(i);
    thetaL_MAP(:,i) = (sum(zzTL,3)+SigmaV/gamma*eye(size(zL,1)))^-1*sum(repmat(y,size(zL,1),1).*zL,2);
    thetaQ_MAP(:,i) = (sum(zzTQ,3)+SigmaV/gamma*eye(size(zQ,1)))^-1*sum(repmat(y,size(zQ,1),1).*zQ,2);
end

%% Plot results - MAP: variation with gamma
clrs = lines(length(params));
figure('units','normalized','outerposition',[0.01 0.04 0.99 0.95]);

ax=subplot(121); hold on; box on; ax=gca; ax.XScale = 'log';
axis([gammaArray(1) gammaArray(end) min([params;thetaL_MAP(:);thetaQ_MAP(:)])-0.5 ...
    max([params;thetaL_MAP(:);thetaQ_MAP(:)])+1]);
p11=plot(gammaArray,repmat(params,1,length(gammaArray)),'--','LineWidth',2); 
xlabel('gamma'); ylabel('parameters'); title('True parameters');
lgnd=legend([p11],'c','b(1)','b(2)','A(1,1)','A(1,2)+A(2,1)','A(2,2)');
lgnd.Location = 'north'; lgnd.Orientation = 'horizontal'; lgnd.NumColumns = 3; box(lgnd,'off');
pause;

set(gca,'ColorOrderIndex',1); p12=plot(gammaArray,thetaL_MAP,'-','LineWidth',2);
title('MAP Parameter estimation: linear model');
lgnd=legend([p11],'c','b(1)','b(2)','A(1,1)','A(1,2)+A(2,1)','A(2,2)');
pause;

%% Function to calculate y (without noise), given x and parameters
function y = yFunc(x,T,A,b,c)
    y =  diag(x'*A*x)' + b'*x + c;
end
