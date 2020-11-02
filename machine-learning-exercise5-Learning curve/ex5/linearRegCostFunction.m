function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_theta_x = X * theta;
sum = 0;
for i = 1:m
  sum += (h_theta_x(i, 1) - y(i, 1))^2;
endfor
sum = sum / 2 / m;
J = sum;
res = 0;
for j = 2:size(theta, 1)
  res += theta(j, 1)^2;
endfor
res = res * lambda / 2 / m;
J += res;


for j = 1:size(theta, 1)
  sum = 0;
  for i = 1:m
    sum += (h_theta_x(i, 1) - y(i, 1)) * X(i, j);
  endfor
  sum /= m;
  grad(j, 1) = sum;
endfor

for j = 2:size(theta, 1)
  grad(j, 1) += theta(j, 1) * lambda / m;
endfor



% =========================================================================

grad = grad(:);

end
