function y = lwlr(X_train, y_train, x, tau)

%%% YOUR CODE HERE
lambda = 0.0001;
w = zeros(size(y_train,1),1);

for i=1:size(y_train,1)
   w(i) = exp(-(X_train(i,:)*x)^2/(2*tau*tau));
end
X_train = [ones(size(X_train(:,1)),1) X_train];
l = zeros(size(X_train,2),1);
theta = zeros(size(X_train(1,:),2),1);
l = X_train' * (w .* sigmoid(X_train*theta) .* (y_train - sigmoid(X_train*theta))) - lambda * theta;
h = zeros(size(theta),size(theta));
d = zeros(size(y_train(:,1),1),size(y_train(:,1),1));
for i = 1:size(d,1)
d(i,i) = -w(i) * sigmoid(X_train(i,:)*theta) * (1 - sigmoid(X_train(i,:)*theta));
end
h = X_train' * d * X_train - lambda;
l_old = zeros(size(X_train,2),1);
while((l_old - l)' * (l_old - l) > 0.0000000000000000000000000005)
theta = theta - pinv(h) * l;
l_old = l;
l = X_train' * (w .* sigmoid(X_train*theta) .* (y_train - sigmoid(X_train*theta))) - lambda * theta;
end

if sigmoid(theta' * [1;x]) > 0.5,
  y = 1;
else,
y = 0;
end

end