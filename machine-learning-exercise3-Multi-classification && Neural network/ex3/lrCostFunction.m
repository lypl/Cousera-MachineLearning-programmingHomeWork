function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% y: 5000 * 10; X: 5000 * 400 == h_theta_x: 5000 * 10(m * s); J(theta): 1 * 10; grad: 400 * 10 == theta(10 classifier)
% note: Iterate each classifier training, so theta is still a column, so J is a number rather a matrix

h_theta_x = sigmoid(X * theta);
one_mines_h_theta_x = 1 - h_theta_x;
one_mines_y = 1 - y;
fi = (-y).*log(h_theta_x);
sd = one_mines_y.*log(one_mines_h_theta_x);
J = sum(fi - sd);
J = J / m;
res = (sum(theta.^2) - theta(1, 1) * theta(1, 1)) * lambda / 2 / m; % remember minus theta0 ^ 2; not sure multiple theta0
J = J + res; % point: need to regularized! and need to minus (theta0 ^ 2)j = 0


grad = (X') * (h_theta_x - y) / m; % find the ex3 P6
% new_theta = [zeros(1, size(theta, 2));theta(2:size(theta, 1))]; % theta0 not need regularization
% grad = grad.+(new_theta * lambda / m);
 new_theta = [zeros(1, 1);theta(2:size(theta, 1), :)];
 grad = grad.+(new_theta * lambda / m);

% just one column theta is ok!!!! cuz: oneVsAll.m is iterate for every classifier






% =============================================================

grad = grad(:);

end
