function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m, 1) X];
a2 = sigmoid(a1 * (Theta1'));
a2 = [ones(m, 1) a2];
a3 = sigmoid(a2 * (Theta2')); % m * K

new_y = zeros(m, num_labels); % this format y is necessary
for i = 1:m
  pos = y(i, 1);
  new_y(i, pos) = 1;
endfor

for i = 1:m
  for k = 1:num_labels
    res = -new_y(i, k) * log(a3(i, k)) - (1 - new_y(i, k)) * log(1 - a3(i, k));
    J += res;
  endfor
endfor
J = J / m;

% submit here to get the first score

sum = 0;
for j = 1:size(Theta1, 1)
  for k = 2:size(Theta1, 2)
    sum = sum + Theta1(j, k)^2;
  endfor
endfor

for j = 1:size(Theta2, 1)
  for k = 2:size(Theta2, 2)
    sum = sum + Theta2(j, k)^2;
  endfor
endfor
sum = sum * lambda / 2 / m;

J = J + sum;

% cal gradient

delta_3 = zeros(num_labels, 1);
delta_2 = zeros(size(Theta1, 1) + 1, 1);

DELTA_2 = zeros(size(Theta2));
DELTA_1 = zeros(size(Theta1));
for i = 1:m
  % feedforward
  
  z_2 = a1(i, :) * (Theta1');
  a_2 = sigmoid(z_2);
  a_2 = [ones(size(a_2, 1), 1) a_2];
  z_3 = a2(i, :) * (Theta2');
  a_3 = sigmoid(z_3);
  
  % debug here
  % size(z_2)
  
  a_1 = a1(i, :)';
  z_2 = z_2';
  a_2 = a_2';
  z_3 = z_3';
  a_3 = a_3'; % all k * 1 matrix
  
  
  delta_3 = a_3 - (new_y(i, :)'); % new_y(i, :) need to be k*1
%  fprintf('delta_3 size: \n');
%  size(delta_3)
  delta_2 = ((Theta2') * delta_3);
  delta_2 = delta_2(2:end); % change size of delta_2 first, than do .* operation
  delta_2 = delta_2.*sigmoidGradient(z_2);
  DELTA_2 = DELTA_2 + delta_3 * (a_2');
  DELTA_1 = DELTA_1 + delta_2 * (a_1');
endfor

for i = 1:size(Theta1_grad, 1)
  for j = 1:size(Theta1_grad, 2)
    Theta1_grad(i, j) = DELTA_1(i, j) / m;
  endfor
endfor


for i = 1:size(Theta2_grad, 1)
  for j = 1:size(Theta2_grad, 2)
    Theta2_grad(i, j) = DELTA_2(i, j) / m;
  endfor
endfor

% submit for not regularization

for i = 1:size(Theta1_grad, 1)
  for j = 2:size(Theta1_grad, 2)
    Theta1_grad(i, j) += Theta1(i, j) * lambda / m;
  endfor
endfor


for i = 1:size(Theta2_grad, 1)
  for j = 2:size(Theta2_grad, 2)
    Theta2_grad(i, j) += Theta2(i, j) * lambda / m;
  endfor
endfor







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
