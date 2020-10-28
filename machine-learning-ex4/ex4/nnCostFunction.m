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
Delta_1= zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));

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
% Add ones to the X data matrix
X = [ones(m, 1) X];
% Transform y to 10 dimension
Y = zeros(m,num_labels);
for i=1:m
    Y(i,y(i))= 1;
end

% Part 1 : Compute the cost function
% Compute the hidden layer
a_1 = X;
a_2 = sigmoid(a_1 * Theta1'); % 5000*25 matrice
% Compute the output layer
a_2 = [ones(m, 1) a_2];
h_theta = sigmoid(a_2 * Theta2'); % 5000*10 matrice
% Compute the cost function over all examples
J = sum(sum(-Y.*log(h_theta)-(1-Y).*log(1-h_theta)))/m ...
+(sum(sum(Theta1(:,2:end).^2))+ sum(sum(Theta2(:,2:end).^2)))*lambda/(2*m);


% Part 2 : Compute the gradients without regularization
Delta_2 = zeros(size(Theta2));
Delta_1 = zeros(size(Theta1));
for t=1:m
% Step 1: Compute the activations of each layer
    a1 = X(t,:)'; % add the bias unit
    z2 = Theta1 * a1;
    a2 = [1 ; sigmoid(z2)]; % add the bias unit
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
% Step 2: Compute error for each output unit k in the output layer
    delta_3 = a3 - Y(t,:)'; 
    
    
% Step 3: Compute error for each node in the hidden layer
    delta_2 = Theta2(:,2:end)'*delta_3.*sigmoidGradient(z2);
    
% Step 4: Accumulate the gradient for every example
    Delta_2 = Delta_2 + delta_3*a2';
    Delta_1 = Delta_1 + delta_2*a1';
end

% Step 5: Obtain the gradient for the neural network cost function
Theta1_grad = Delta_1/m;
Theta2_grad = Delta_2/m;


% Part 3: Compute the gradients with regularization
Theta1_grad(:,1) = Delta_1(:,1)/m;
Theta1_grad(:,2:end) = Delta_1(:,2:end)/m + Theta1(:,2:end)*lambda/m;

Theta2_grad(:,1) = Delta_2(:,1)/m;
Theta2_grad(:,2:end) = Delta_2(:,2:end)/m + Theta2(:,2:end)*lambda/m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
