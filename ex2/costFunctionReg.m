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

% --- COST FUNCTION ---
for i = 1:m

	% Compute hypothesis
	hypothesis = sigmoid(theta' * X(i,:)'); 

	% Add to cost function
	J += ((-y(i, 1) * log(hypothesis) - ((1 - y(i, 1)) * log(1 - hypothesis) )));

end

J = J / m;		% Divide by number of training sets

jTemp = 0;

for j = 2:size(theta)
	jTemp += theta(j) ^ 2;
end

jTemp = (jTemp * lambda) / (2*m);

J += jTemp;

% --- GRADIENT DESCENT --- %

temp = zeros(size(theta), 1);


for i = 1:m
	hypothesis = sigmoid(theta' * X(i,:)');

	% jth = 0
	temp(1) += (((hypothesis) - y(i)) * X(i, 1)) / m;

	% jth >= 1
	for j = 2:size(theta)
		temp(j) += ((((hypothesis) - y(i)) * X(i, j)) / m);
	end
end	

for j = 2:size(theta)
	temp(j) += (lambda / m) * theta(j);
end

for i = 1:size(grad)
	grad(i) = temp(i);
end




% =============================================================

end
