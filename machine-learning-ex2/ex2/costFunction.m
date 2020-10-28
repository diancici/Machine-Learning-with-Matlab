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
% Compute cost J
for i=1:m
    h_theta = sigmoid(X(i,:)* theta);
    J = J-y(i)*log(h_theta)-(1-y(i))*log(1-h_theta);  
end
J = J/m;

% Compute gradient
h_theta = sigmoid(X(:,:)* theta);
dif = (h_theta - y(:,:)).* X(:,:); % m*(n+1) matrice
dif_sum = sum(dif); % 1*(n+1) vector
grad = dif_sum/m;  
% =============================================================

end
