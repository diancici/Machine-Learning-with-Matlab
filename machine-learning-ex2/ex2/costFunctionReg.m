function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); % n*1 vector
reg = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
for i=1:m
    h_theta = sigmoid(X(i,:)* theta);
    J = J-y(i)*log(h_theta)-(1-y(i))*log(1-h_theta); 
    grad = grad + ((h_theta - y(i)).* X(i,:))'/m; % n*1 vector
end

for j=2:size(theta)
    reg = reg + lambda*theta(j).^2/2;
    grad(j) = grad(j)+ lambda*theta(j)/m; % compute the gradient
end

% Compute the cost function
J = (J+reg)/m;





% =============================================================

end
