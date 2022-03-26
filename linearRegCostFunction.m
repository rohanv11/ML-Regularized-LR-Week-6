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

% X: m * n + 1
% y: m * 1
% theta: (n + 1) by 1                      
% hypothesis1: X * theta = m * 1

hypothesis1 = X * theta
% hypothesis1: m * 1

regularization_term = lambda / (2 * m) * sum(theta(2:size(theta, 1), 1) .^ 2);



J = 1 / (2 * m) * sum((hypothesis1 - y) .^ 2)  +  regularization_term;

% hypothesis1 - y: m by 1
% X' = (n + 1) by m
% X' * (hypothesis1 - y): (n + 1) by 1

% X: m by (n + 1)
% (hypothesis1 - y)': 1 by m
% (hypothesis1 - y)' * X: 1 by (n + 1)

grad(1, 1) = (1 / m) * ((hypothesis1 - y)' * X )(1,1);

temp = ((hypothesis1 - y)' * X )
temp = reshape(temp, size(temp, 2) , size(temp, 1))


grad(2:end, 1) = (1 / m) * temp(2:end, 1) +  (lambda / m) * theta(2:end, 1);

%(X' * (hypothesis1 - y))(2:end, 1): n by 1
%theta(2:end, 1): n by 1 






% =========================================================================

%grad = grad(:);

end
