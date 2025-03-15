%load the previous model
load('multiclass_perc_model.mat','weights','biases');
disp(['Model loaded.']); %insurance feedback

trial_digits = {
    [0 0 1 0 0; 0 1 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 1 1 1 1 1],  % 1  %1
    [1 1 1 1 1; 0 0 0 0 0; 0 0 0 0 1; 0 1 1 1 1; 0 0 0 0 0; 0 0 0 0 1; 1 1 1 1 1],  % 3  %2
    [1 0 0 0 1; 1 0 0 0 0; 1 0 0 0 1; 1 1 0 1 1; 0 0 0 0 1; 0 0 0 0 1; 0 0 0 0 1],  % 4  %3
    [1 1 1 1 1; 1 0 0 0 0; 1 0 0 0 0; 1 1 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1],  % 5  %4
    [1 0 1 1 1; 0 0 0 0 1; 0 0 0 0 1; 0 1 0 0 1; 0 0 0 0 1; 0 0 0 0 1; 0 0 0 0 1],  % 7  %5
    [1 1 1 1 1; 1 0 0 0 1; 1 0 1 0 1; 1 1 1 1 1; 1 0 1 0 1; 1 0 0 0 1; 1 1 1 1 1],  % 8  %6
    [1 1 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1],  % 0  %7
    [1 1 1 1 1; 0 0 0 0 0; 0 0 0 0 1; 1 1 0 1 1; 1 0 0 0 0; 1 0 0 0 0; 1 1 1 1 1]   % 2  %8
};

% a function to recognise the digits using our perceptron
function recognise_digits(trial_digits,weights,biases)
    for a=1:length(trial_digits)
        %take a digit from my matrix of noisy digits
        trial_digit = trial_digits{a};    %the trial digits will be given in a cell for ease of code 
        digit_vector = reshape(trial_digit,[],1);
        %perceptron output:
        output = weights * digit_vector + biases;
        output = output >= 0;
        [~,predicted_class] = max(output) ; %get the index of max value--same thing as the training code 
        %show the recognised digit
        disp(['Given matrix: ', num2str(a), ', Recognised digit: ', num2str(predicted_class - 1)]);
    end     
end

recognise_digits(trial_digits,weights,biases);

