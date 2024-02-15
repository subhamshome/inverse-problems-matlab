% Clear workspace, close all figures, and clear command history
clear, close all;
clc;

% Load DataOne and DataTwo from mat files
Data1 = load('DataOne.mat');
Data2 = load('DataTwo.mat');

% Display the data using showData function
showData(Data1, Data2);

% Display the frequency images using frequency_image function
frequency_image(Data1, "Image 1");
frequency_image(Data2, "Image 2");

% Display the transfer functions using transfer_function function
transfer_function(Data1, "Image 1");
transfer_function(Data2, "Image 2");

% Set the value of mus
mus = 0.1;

% Calculate inverse Fourier transform using cal_ifft function
cal_ifft(Data1, mus, "Image 1");
cal_ifft(Data2, mus, "Image 2");

% Calculate the distance using getDistance function
getDistance(Data1, "Image 1");
getDistance(Data2, "Image 2");

% Function to display the original data
function showData(Data1, Data2)
figure('Name', 'Original Data'), clf;
set(gcf, 'WindowState', 'maximized');
subplot(2, 3, 1), showImage(Data1.TrueImage, 'True Image 1');
subplot(2, 3, 2), showImage(Data1.Data, 'Data 1');
subplot(2, 3, 3), showImage(Data1.IR, 'IR 1');
subplot(2, 3, 4), showImage(Data2.TrueImage, 'True Image 2');
subplot(2, 3, 5), showImage(Data2.Data, 'Data 2');
subplot(2, 3, 6), showImage(Data2.IR, 'IR 2');
sgtitle("Details of Image 1 and Image 2");
saveas(gcf, 'fig1-intro.png');
end

% Function to display an image
function showImage(image, titleText)
imagesc(image);
colormap('gray');
colorbar;
axis square off;
title(titleText);
end

% Function to display the frequency subplots
function fftSubplots(fft_true, fft_data, name)
axis_v = linspace(-0.5, 0.5, 256);
figure('Name',name), clf;
set(gcf, 'WindowState', 'maximized');
subplot(231), showFFT(abs(fft_true), axis_v, 'Magnitude');
subplot(232), showFFT(log(abs(fft_true)), axis_v, 'Magnitude log');
subplot(233), showFFT(angle(fft_true), axis_v, 'Phase');
subplot(234), showFFT(abs(fft_data), axis_v, 'Magnitude');
subplot(235), showFFT(log(abs(fft_data)), axis_v, 'Magnitude log');
subplot(236), showFFT(angle(fft_data), axis_v, 'Phase');
sgtitle(name + newline + "Top row: True Image, Bottom row: Data");
saveas(gcf, name+'_fig_freq.png');
end

% Function to display the frequency images
function frequency_image(Data, name)
fft_data = MyFFT2(Data.Data);
fft_true = MyFFT2(Data.TrueImage);
fftSubplots(fft_true, fft_data, name);
end

% Function to display the FFT plot
function showFFT(data, axis_v, titleText)
imagesc(axis_v, axis_v, data);
colormap('gray');
colorbar;
axis square;
title(titleText);
end

% Function to display the transfer function
function transfer_function(Data, name)
axis_v = linspace(-0.5, 0.5, 256);
transfer = abs(MyFFT2RI(Data.IR, 256));

figure('Name', name + " IR"), clf;
set(gcf, 'WindowState', 'maximized');
subplot(221), showImage(Data.IR, 'IR');
subplot(222), showImage(transfer, 'Transfer function');
subplot(223), plot(axis_v, transfer(128, :)), title('Slice at 0');
h = subplot(224);
mesh(axis_v, axis_v, transfer), title('3D filter');
colormap(h, "turbo");
sgtitle(name + " IR");
saveas(gcf, name + "_fig_transfer.png");
end

% Function to perform deconvolution
function [x, x_hat] = deconv(obs, IR, mu)
h = MyFFT2RI(IR, 256);

d1 = [[0  0 0]
    [0 -1 1]
    [0  0 0]];
d2 = [[0  0 0]
    [0 -1 0]
    [0  1 0]];
d = MyFFT2RI(d1, 256) + MyFFT2RI(d2, 256);

g = conj(h) ./ (abs(h) .^ 2 + mu * abs(d) .^ 2);
y = MyFFT2(obs);
x_hat = g .* y;
x = MyIFFT2(x_hat);
end

% Main function to calculate the inverse Fourier transform
function cal_ifft(Data, mus, name)
axis_v = linspace(-0.5, 0.5, 256);
if length(mus) > 1
    figure('Name', 'Spatial domain'), clf;
    set(gcf, 'WindowState', 'maximized');
    subplot(ceil((length(mus) + 1) / 3), 3, 1), showImage(Data.TrueImage, 'True image');
    subplot(ceil((length(mus) + 1) / 3), 3, 2), showImage(Data.Data, 'Observed image');
    for i = 1:length(mus)
        subplot(ceil((length(mus) + 1) / 3), 3, i + 2);
        [reconstructed_img, ~] = deconv(Data.Data, Data.IR, mus(i));
        showImage(reconstructed_img, sprintf("mu=%2.2f", mus(i)));
    end
    sgtitle(name + ": Spatial Domain");
    saveas(gcf, name + "_fig_ifft_spatial.png")
    
    figure('Name', 'Frequency domain'), clf;
    set(gcf, 'WindowState', 'maximized');
    subplot(ceil((length(mus) + 1) / 3), 3, 1), showFFT(log(abs(MyFFT2(Data.TrueImage))), axis_v, 'True image');
    subplot(ceil((length(mus) + 1) / 3), 3, 2), showFFT(log(abs(MyFFT2(Data.Data))), axis_v, 'Observed image');
    
    for i = 1:length(mus)
        subplot(ceil((length(mus) + 1) / 3), 3, i + 2);
        [~, reconstructed_img_hat] = deconv(Data.Data, Data.IR, mus(i));
        showFFT(log(abs(reconstructed_img_hat)), axis_v, sprintf("mu=%2.2f", mus(i)));
    end
    sgtitle(name + ": Frequency Domain");
    saveas(gcf, name + "_fig_ifft_freq.png");
    
elseif length(mus) == 1
    [reconstructed_img, ~] = deconv(Data.Data, Data.IR, mus);
    [~, reconstructed_img_hat] = deconv(Data.Data, Data.IR, mus);
    
    figure('Name','Mu 0'), clf;
    set(gcf, 'WindowState', 'maximized');
    subplot(231), showImage(Data.TrueImage, 'True image');
    subplot(232), showImage(Data.Data, 'Observed image');
    subplot(233); showImage(reconstructed_img, sprintf("mu=%2.2f", mus));
    subplot(234), showFFT(log(abs(MyFFT2(Data.TrueImage))), axis_v, 'True image');
    subplot(235), showFFT(log(abs(MyFFT2(Data.Data))), axis_v, 'Observed image');
    subplot(236); showFFT(log(abs(reconstructed_img_hat)), axis_v, sprintf("mu=%2.2f", mus));
    
    sgtitle(name + ": Data Analysis at mu=" + mus + newline + "Top: Spatial Domain, Bottom: Frequency Domain");
    saveas(gcf, name + "_fig_ifft_freq_mu0.png");
end
end

% Functions for various norms
function d = d_1(reconstructed_img, true)
d = sum(abs(reconstructed_img - true), "all") / sum(abs(true), "all");
end

function d = d_2(reconstructed_img, true)
d = sum((reconstructed_img - true).^2, "all") / sum(true.^2, "all");
end

function d = d_inf(reconstructed_img, true)
d = max(abs(reconstructed_img - true),[],'all') / max(abs(true),[],'all');
end

% Function to calculate the distance
function getDistance(Data, name)
log_mu_spacing = -10:10;
d_1_values = zeros(length(log_mu_spacing), 1);
d_2_values = zeros(length(log_mu_spacing), 1);
d_inf_values = zeros(length(log_mu_spacing), 1);

for i = 1:length(log_mu_spacing)
    [reconstructed_img, ~] = deconv(Data.Data, Data.IR, 10^log_mu_spacing(i));
    d_1_values(i) = d_1(reconstructed_img, Data.TrueImage);
    d_2_values(i) = d_2(reconstructed_img, Data.TrueImage);
    d_inf_values(i) = d_inf(reconstructed_img, Data.TrueImage);
end

figure('Name', 'Distances for different mu values'), clf;
set(gcf, 'WindowState', 'maximized');
subplot(131), plotDistances(log_mu_spacing, d_1_values, 'mu');
subplot(132), plotDistances(log_mu_spacing, d_2_values, 'mu');
subplot(133), plotDistances(log_mu_spacing, d_inf_values, 'mu');
sgtitle(name + ": Distance Values");
saveas(gcf, name + "_fig_dist_mu.png");
end

% Function to plot the distance values
function plotDistances(x, y, xLabel)
hold on;
plot(x, y);
[~, min_idx] = min(y);
[max_v, ~] = max(y);
plot([x(min_idx), x(min_idx)], [0, max_v]);
xlabel('Logarithmic Spacing');
ylabel('Distance Values');
title(sprintf("%s = %f", xLabel, 10^x(min_idx)));
end
