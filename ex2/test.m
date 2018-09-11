function y = test(x)

p = [1 2; 4 5];

if (sigmoid(0.2) >= 0.5)
	p(1,2) = 222;
end

disp(p);