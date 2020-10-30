function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

h_theta_x = sigmoid(X * theta); % size: m * 1
for i = 1:m
  res = -y(i, 1) * log(h_theta_x(i, 1)) - (1 - y(i, 1)) * log(1 - h_theta_x(i, 1));
  J = J + res;
endfor
J = J / m;

sz = size(grad, 1);
for j = 1:sz
  res = 0;
  for i = 1:m
    res = res + (h_theta_x(i, 1) - y(i, 1)) * X(i, j);
  endfor
  res = res / m;
  grad(j, 1) = res;
endfor




% =============================================================

end
