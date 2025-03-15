clear all
% Step 1: Define binary matrices for each digit (0-9) as 7x5 matrices
d0= [1 1 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1];  % 0
d1= [0 0 1 0 0; 0 1 1 0 0; 1 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 1 1 1 1 1];  % 1
d2= [1 1 1 1 1; 0 0 0 0 1; 0 0 0 0 1; 1 1 1 1 1; 1 0 0 0 0; 1 0 0 0 0; 1 1 1 1 1];  % 2
d3= [1 1 1 1 1; 0 0 0 0 1; 0 0 0 0 1; 0 1 1 1 1; 0 0 0 0 1; 0 0 0 0 1; 1 1 1 1 1];  % 3
d4= [1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1; 0 0 0 0 1; 0 0 0 0 1; 0 0 0 0 1];  % 4
d5= [1 1 1 1 1; 1 0 0 0 0; 1 0 0 0 0; 1 1 1 1 1; 0 0 0 0 1; 0 0 0 0 1; 1 1 1 1 1];  % 5
d6= [1 1 1 1 1; 1 0 0 0 0; 1 0 0 0 0; 1 1 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1];  % 6
d7= [1 1 1 1 1; 0 0 0 0 1; 0 0 0 0 1; 0 0 0 0 1; 0 0 0 0 1; 0 0 0 0 1; 0 0 0 0 1];  % 7
d8= [1 1 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1];  % 8
d9= [1 1 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1; 0 0 0 0 1; 0 0 0 0 1; 1 1 1 1 1];  % 9

v0 = d0(:);
v1 = d1(:);
v2 = d2(:);
v3 = d3(:);
v4 = d4(:);
v5 = d5(:);
v6 = d6(:);
v7 = d7(:);
v8 = d8(:);
v9 = d9(:);
% Step 2: Flatten each matrix to a column vector (35x1)

inputs = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9];

% Step 3: Define target matrix (1-of-10 encoding for digits 0 to 9)
targets = eye(10);

% Step 4: Initialize weights and biases for single-layer perceptron
num_inputs = size(inputs,1);  % 35 pixels per digit
num_outputs = size(targets,1); % 10 possible digits (0 to 9)
weights = rand(num_outputs, num_inputs) * 0.01;  % Random weights
biases = rand(num_outputs, 1) * 0.01;            % Random biases
learning_rate = 0.1;
num_epochs = 10;

% Step 5: Training loop
for epoch = 1:num_epochs
    % For each training example
    for i = 1:10
        t = targets(:, i);             % Target vector (1-hot encoding)
        % Compute the perceptron output
        y = weights * inputs(:, i) + biases;      % Linear combination
        y = y >= 0;                    % Threshold activation (binary output)

        % Update weights and biases (only if there is an error)
        error = t - y;                 % Calculate error
        weights = weights + learning_rate * (error * inputs(:,i)'); % Weight update
        biases = biases + learning_rate .* error;          % Bias update
    end
end

% Step 6: Testing the trained perceptron


disp('Testing the trained perceptron with corrupted digits:');
corrupt_1 = [0 0 1 0 0; 0 1 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 1 1 1 1 1];  % 1 
corrupt_3 = [1 1 1 1 1; 0 0 0 0 0; 0 0 0 0 1; 0 1 1 1 1; 0 0 0 0 0; 0 0 0 0 1; 1 1 1 1 1];  % 3  
corrupt_4 = [1 0 0 0 1; 1 0 0 0 0; 1 0 0 0 1; 1 1 0 1 1; 0 0 0 0 1; 0 0 0 0 1; 0 0 0 0 1];  % 4  
corrupt_5 = [1 1 1 1 1; 1 0 0 0 0; 1 0 0 0 0; 1 1 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1];  % 5  
corrupt_7 = [1 0 1 1 1; 0 0 0 0 1; 0 0 0 0 1; 0 1 0 0 1; 0 0 0 0 1; 0 0 0 0 1; 0 0 0 0 1];  % 7  
corrupt_8 = [1 1 1 1 1; 1 0 0 0 1; 1 0 1 0 1; 1 1 1 1 1; 1 0 1 0 1; 1 0 0 0 1; 1 1 1 1 1];  % 8  
corrupt_0 = [1 1 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1];  % 0  
corrupt_2 = [1 1 1 1 1; 0 0 0 0 0; 0 0 0 0 1; 1 1 0 1 1; 1 0 0 0 0; 1 0 0 0 0; 1 1 1 1 1];  % 2 
vc1 = corrupt_1(:);
vc3 = corrupt_3(:);
vc4 = corrupt_4(:);
vc5 = corrupt_5(:);
vc7 = corrupt_7(:);
vc8 = corrupt_8(:);
vc0 = corrupt_0(:);
vc2 = corrupt_2(:);
corrupt_inputs = [vc1 vc3 vc4 vc5 vc7 vc8 vc0 vc2];
correct_predictions_corr = 0; %variable that contains our correct predictions
for i = 1:8
    y_corr = weights * corrupt_inputs(:, i) + biases;      % Linear combination
    y_corr = y_corr >= 0;                    % Threshold activation (binary output)
    %Find the index of the maximum value of output to determin predicted class
    %maximize the output of a neural network across all predictions i.e. the output values of the last layer. 
    [~,predicted_class_corr] = max(y_corr);
    %count correct predictions
    if predicted_class_corr - i == i
        correct_predictions_corr = correct_predictions_corr +1;
    end
    disp(['Given matrix:',num2str(i), ' Recognised:', num2str(predicted_class_corr - 1)]);
end

%calculate accuracy 
% accuracy = (correct_predictions_corr/10) * 100; %accuracy percentages
% disp(['Corrputed Data Accuracy: ',num2str(accuracy),'%']);