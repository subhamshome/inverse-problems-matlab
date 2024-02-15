% Clear workspace, close all figures, and clear command history
clear, close all;
clc;

Data1 = load('DataOne.mat');
Data2 = load('DataTwo.mat');

%%
T = 0.2; % Threshold
alpha = 0.045; %0.45 with smaller alpha it will converge slower
num_iter = 3000; %300
mu = 0.1;
data = Data1;

start_value = 0.1;
end_value = 0.5;
step_size = 0.005;
num_points = floor((end_value - start_value) / step_size) + 1;
vector = linspace(0.1, 0.5, num_points);
%%

alpha_values = vector;
elapsed_times = zeros(size(alpha_values));


for i = 1:length(alpha_values)
    alpha = alpha_values(i);
    
    % Measure the time taken for the calculate function to execute
    tic;
    calculate(data, T, alpha, num_iter, mu)
    elapsed_times(i) = toc;
    
    fprintf('Elapsed time for alpha = %.5f: %.4f seconds\n', alpha, elapsed_times(i));
end
%%
% Generate a finer grid of alpha values for interpolation
alpha_fine = linspace(min(alpha_values), max(alpha_values), 10*numel(alpha_values));

% Interpolate elapsed times on the finer grid
elapsed_times_fine = interp1(alpha_values, elapsed_times, alpha_fine, 'spline');

fh = figure('WindowState', 'maximized');
% set(gcf, 'windowsize')
plot(alpha_fine, elapsed_times_fine, 'LineWidth', 1.15);
[min_elapsed_time, min_index] = min(elapsed_times);
hold on;
% plot(alpha_values(min_index), min_elapsed_time, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
xlabel('Alpha Values');
ylabel('Elapsed Time (seconds)');
title('Elapsed Time for Different Alpha Values');
% subtitle(['Min Time: ' num2str(min_elapsed_time) 's, Min Alpha: ' num2str(alpha_values(min_index))]);
legend('Elapsed Time');
filename = [num2str(mu), '_alpha_vs_elapsedTime.png'];
saveas(gcf, filename);
%%
% % testing for different alpha values, smaller value will make 
% % the convergence slower. Values > 1 will diverge
% 
tic;
calculate(data, T, 0.25, num_iter, 0.1)
% % calculate(data, T, 0.05, num_iter, mu)
% % calculate(data, T, 0.005, num_iter, mu)
% tic;
% calculate(data, T, 1e-5, num_iter, mu)
elapsed_time = toc;

%% Funtion to do the Image Restoration
function calculate(Data, T, alpha, num_iter, mu)
    % calculate - Perform image reconstruction using a modified total variation method.
    %
    %   This function takes input parameters and iteratively reconstructs an
    %   image by minimizing a cost function that incorporates a modified total
    %   variation (TV) regularization term.
    %
    %   Parameters:
    %       Data (struct): Struct containing image data and relevant information.
    %           - Data.Data: Observed image data
    %           - Data.IR: Point spread function or impulse response
    %           - Data.TrueImage: Ground truth or true image
    %       T (double): Threshold parameter for the TV regularization
    %       alpha (double): Weighting factor for the TV regularization term
    %       num_iter (int): Number of iterations for the reconstruction process
    %       mu (double): Regularization parameter for controlling the trade-off
    %                    between data fidelity and TV regularization
    %
    %   Output:
    %       The function generates visualizations and displays the reconstructed
    %       image, original image, observed image, and relevant metrics.
    
    % Get the size of the data
    [dataHeight, dataWidth] = size(Data.Data);
    
    % Define constants
    mu_p = mu / alpha;
    mse = zeros(num_iter, 1);
    
    % Difference matrices
    differenceMatrixCol = [0, 0, 0; 0, -1, 1; 0, 0, 0];
    differenceMatrixRow = [0, 0, 0; 0, -1, 0; 0, 1, 0];
    
    % FFT of the observed image
    observedImageFFT = MyFFT2(Data.Data);
    
    % FFT of the difference matrices row and column-wise
    differenceMatrixColFFT = MyFFT2RI(differenceMatrixCol, dataHeight);
    differenceMatrixRowFFT = MyFFT2RI(differenceMatrixRow, dataWidth);
    
    % Interpixel differences row and column-wise
    deltaCol = MyIFFT2(differenceMatrixColFFT .* observedImageFFT);
    deltaRow = MyIFFT2(differenceMatrixRowFFT .* observedImageFFT);
    
    % Update auxiliary variables row and column-wise
    auxiliaryCol = (1 - 2 * alpha * min(1, T ./ abs(deltaCol))) .* deltaCol;
    auxiliaryRow = (1 - 2 * alpha * min(1, T ./ abs(deltaRow))) .* deltaRow;
    
    % Denominator parts of the equation to be minimized
    HSquareFFT = abs(MyFFT2RI(Data.IR, dataHeight)) .^ 2;
    DSquareFFT = abs(differenceMatrixColFFT) .^ 2 + abs(differenceMatrixRowFFT) .^ 2;
    
    % Numerator parts of the equation to be minimized
    HConjFFT = conj(MyFFT2RI(Data.IR, dataHeight));
    DColConjFFT = conj(differenceMatrixColFFT);
    DRowConjFFT = conj(differenceMatrixRowFFT);
    
    auxiliaryColFFT = MyFFT2(auxiliaryCol);
    auxiliaryRowFFT = MyFFT2(auxiliaryRow);
    
    % Initialize variables
    processedData = Data.Data;
    % previousData = zeros(size(Data.Data));
    
    for iteration = 1:num_iter
        % Calculate the components for the minimization equation
        xDenom = HSquareFFT + mu_p * DSquareFFT;
        xNum = HConjFFT .* observedImageFFT + ...
            mu_p * DColConjFFT .* auxiliaryColFFT + ...
            mu_p * DRowConjFFT .* auxiliaryRowFFT;
        
        % Minimization equation
        xFFT = xNum ./ xDenom;
        
        % Update interpixel differences
        deltaCol = MyIFFT2(differenceMatrixColFFT .* xFFT);
        deltaRow = MyIFFT2(differenceMatrixRowFFT .* xFFT);
        
        % Update auxiliary variables
        auxiliaryCol = (1 - 2 * alpha * min(1, T ./ abs(deltaCol))) .* deltaCol;
        auxiliaryRow = (1 - 2 * alpha * min(1, T ./ abs(deltaRow))) .* deltaRow;
        
        auxiliaryColFFT = MyFFT2(auxiliaryCol);
        auxiliaryRowFFT = MyFFT2(auxiliaryRow);
        
        % Update variables
        previousData = processedData;
        processedData = MyIFFT2(xFFT);
        
        % Calculate Mean Squared Error
        mse(iteration) = calculateNormalizedSquaredError(processedData, ...
            Data.TrueImage);
        
        % Check for convergence
        if calculateNormalizedSquaredError(processedData, previousData) < 1e-14
            break;
        end
    end
    
    % Calculate the iteration index of convergence
    lastIteration = max(tril(find(mse ~= 0)), [], 1);

    % Plots
    figure("Name",sprintf("alpha = %.3f", alpha));
    set(gcf, 'WindowState', 'maximized');

    subplot(231);
    imagesc(Data.TrueImage);
    title('True Image');
    colormap('gray'); colorbar; axis('square');

    subplot(232)
    imagesc(Data.Data);
    title('Observed Image');
    colormap('gray'); colorbar; axis('square');

    subplot(233)
    imagesc(processedData);
    title('Resconstructed Image');
    colormap('gray'); colorbar; axis('square');

    subplot(234);
    showFFT(log(abs(xFFT)), 'FFT of Image - Log space');

    subplot(235);
    showFFT(angle(xFFT), 'FFT of Image - Phase');

    subplot(236);
    plot(mse(mse > 0));
    title('d_2 error');

    subplot_title = ['Threshold=', num2str(T), ', mu=', num2str(mu), ...
        ', alpha=', num2str(alpha), ', Error=', ...
        num2str(mean(mse)), ', Iterations=', num2str(lastIteration)];
    sgtitle(subplot_title);
    
    % filename=['mu_', num2str(mu), '.png'];
    % saveas(gcf, filename);
end

function normalizedSquaredError = calculateNormalizedSquaredError(reconstructedImage, trueImage)
    % Compute the squared differences between reconstructed and true images
    squaredDifferences = (reconstructedImage - trueImage).^2;

    % Compute the normalized squared error
    normalizedSquaredError = sum(squaredDifferences, 'all') / sum(trueImage.^2, 'all');
end

function showFFT(data, titleText)
    imagesc(data);
    colormap('gray');
    colorbar;
    axis square;
    title(titleText);
end