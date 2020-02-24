% Generate and Plot Training Set 
clear all; close all; 

% Train the model params using 10 samples and
% test it against 10k Samples
[plot_x1,plot_x2] = trainClassifier(10,1);
test_classifier(plot_x1,plot_x2,2);

% Train the model params using 100 samples
[plot_x1,plot_x2] = trainClassifier(100,3);
test_classifier(plot_x1,plot_x2,4);

% Train the model params using 1000 samples
[plot_x1,plot_x2] = trainClassifier(1000,5);
test_classifier(plot_x1,plot_x2,6);

%%  Functions 
function [theta, cost] = gradient_descent(x, N, label, theta, alpha, num_iters)
    cost = zeros(num_iters, 1);
    for i = 1:num_iters % while norm(cost_gradient) > threshold
        h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function   
        cost(i) = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))));
        cost_gradient = (1/N)*(x' * (h - label));
        theta = theta - (alpha.*cost_gradient); % Update theta
    end
end

function cost = cost_func(theta, x, label,N)
    h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function
    cost = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))));
end

function error = plot_test_data(decision, label, Nc, p, fig, x, plot_x1, plot_x2)
    ind00 = find(decision==0 & label==0); % true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % false negative
    ind11 = find(decision==1 & label==1); % true positive
    error = (p10*p(1) + p01*p(2))*100;

    % Plot decisions and decision boundary
    figure(fig);
    plot(x(ind00,1),x(ind00,2),'og'); hold on,
    plot(x(ind10,1),x(ind10,2),'or'); hold on,
    plot(x(ind01,1),x(ind01,2),'+r'); hold on,
    plot(x(ind11,1),x(ind11,2),'+g'); hold on,
    plot(plot_x1, plot_x2);
    axis([plot_x1(1), plot_x1(2), min(x(:,2))-2, max(x(:,2))+2])
    legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions','Classifier');
end


function [plot_x1,plot_x2] = trainClassifier(N,figNumber)
  n = 2;      % number of feature dimensions
  
  % parallel distributions
  mu(:,1) = [-2;0]; Sigma(:,:,1) = [1 -0.9;-0.9  2];
  mu(:,2) = [2;0]; Sigma(:,:,2) = [2 0.9;0.9 1];

  % Class priors for class 0 and 1 respectively
  p = [0.9,0.1]; 

  % Generating true class labels
  label = (rand(1,N) >= p(1))';
  Nc = [length(find(label==0)),length(find(label==1))];

  % Draw samples from each class pdf
  x = zeros(N,n); 
  for L = 0:1
      x(label==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc(L+1));
  end

  %Plot samples with true class labels
  figure(figNumber);
  plot(x(label==0,1),x(label==0,2),'o',x(label==1,1),x(label==1,2),'+');
  legend('Class 0','Class 1'); title('Training Data and True Class Labels');
  xlabel('x_1'); ylabel('x_2'); hold on;

  % Initialize fitting parameters
  x = [ones(N, 1) x(:,1) x(:,2) x(:,1).*x(:,1) x(:,1).*x(:,2) x(:,2).*x(:,2)];
  initial_theta = zeros(n+4, 1);
  label=double(label);

  % Compute gradient descent to get theta values
  [theta, cost] = gradient_descent(x,N,label,initial_theta,1,1000);
  [theta2, cost2] = fminsearch(@(t)(cost_func(t, x, label, N)), initial_theta);

  % Choose points to draw boundary line
  plot_x1 = [min(x(:,2))-2,  max(x(:,2))+2];                      
  plot_x2(1,:) = (-1./theta(3)).*(theta(2).*plot_x1 + theta(1));  
  plot_x2(2,:) = (-1./(theta2(4) + theta2(5).*plot_x1)).*(theta2(3).*plot_x1.*plot_x1 + theta2(2).*plot_x1 + theta2(1)); % fminsearch

  % Plot decision boundary
  plot(plot_x1, plot_x2(2,:));  
  axis([plot_x1(1), plot_x1(2), min(x(:,3))-2, max(x(:,3))+2]);
  legend('Class 0', 'Class 1', 'Classifier (fminsearch)');

end

function test_classifier(plot_x1,plot_x2,figNumber)
  %% Generate test data set 
  N_test = 10000;

  % parallel distributions
  mu(:,1) = [-2;0]; Sigma(:,:,1) = [1 -0.9;-0.9  2];
  mu(:,2) = [2;0]; Sigma(:,:,2) = [2 0.9;0.9 1];

  n=2;

  % Class priors for class 0 and 1 respectively
  p = [0.9,0.1]; 

  % Generating true class labels
  label_test = (rand(1,N_test) >= p(1))';
  Nc_test = [length(find(label_test==0)),length(find(label_test==1))];

  % Draw samples from each class pdf
  x_test = zeros(N_test,n); 
  for L = 0:1
      x_test(label_test==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc_test(L+1));
  end 

  % Coefficients for decision boundary line equation
  coeff(1,:) = polyfit([plot_x1(1), plot_x1(2)], [plot_x2(1,1), plot_x2(1,2)], 1);
  coeff(2,:) = polyfit([plot_x1(1), plot_x1(2)], [plot_x2(2,1), plot_x2(2,2)], 1); %fminsearch
  % Decide based on which side of the line each point is on
  for i = 1:2
      if coeff(i,1) >= 0
          decision(:,i) = (coeff(i,1).*x_test(:,1) + coeff(i,2)) < x_test(:,2);
      else
          decision(:,i) = (coeff(i,1).*x_test(:,1) + coeff(i,2)) > x_test(:,2);
      end
  end

  error2 = plot_test_data(decision(:,2), label_test, Nc_test, p, figNumber, x_test, plot_x1, plot_x2(2,:));
  title('Test Data Classification (using fminsearch)');
  fprintf('Total error (classifier using fminsearch): %.2f%%\n',error2);
end
