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
%j
% Note: grad should have the same dimensions as theta
%

% --- COST FUNCTION ---
for i = 1:m

	% Compute hypothesis
	hypothesis = sigmoid(theta' * X(i,:)'); 

	% Add to cost function
	J += ((-y(i, 1) * log(hypothesis) - ((1 - y(i, 1)) * log(1 - hypothesis) )));

end

J = J / m;		% Divide by number of training sets
 

temp = zeros(size(theta), 1); 	% Temporary values for gradient params

% --- GRADIENT DESCENT ---
for i = 1:m
	hypothesis = sigmoid(theta' * X(i,:)');
	temp(1) += ((hypothesis) - y(i)) * X(i, 1);
	temp(2) += ((hypothesis) - y(i)) * X(i, 2);
	temp(3) += ((hypothesis) - y(i)) * X(i, 3);
end	

grad(1) = temp(1);
grad(2) = temp(2);
grad(3) = temp(3);

grad = grad / m;

% =============================================================

end
