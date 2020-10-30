function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta_x = sigmoid(X * theta);
for i = 1:m
  res = -y(i, 1) * log(h_theta_x(i, 1)) - (1 - y(i, 1)) * log(1 - h_theta_x(i, 1));
  J = J + res;
endfor
J = J / m;

sum = 0;
for i = 2:size(theta, 1) % theta0 is not need to cal
  sum = sum + theta(i, 1) * theta(i, 1);
endfor
J = J + sum * lambda / 2 / m;

for j = 1:size(theta, 1)
  for i = 1:m
    grad(j, 1) = grad(j, 1) + (h_theta_x(i, 1) - y(i, 1)) * X(i, j);
  endfor
  grad(j, 1) = grad(j, 1) / m;
  if j > 1
    grad(j, 1) = grad(j, 1) + theta(j, 1) * lambda / m;
  endif
endfor

% =============================================================

end
