function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
% For a matrix, your function should perform the sigmoid function on every element.

for i = 1:size(z, 1) % size(z) is a matrix, so need to get each length of row and column
  for j = 1: size(z, 2) 
    g(i, j) = 1 / (1 + exp(-z(i, j)));
  endfor
endfor


% =============================================================

end
