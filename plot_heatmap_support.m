clear; clc;

%% Parameters
d = 2;
n = 99;
n_c = 3;
sigma = 8;

n_far = 9;

epsilon = 0.01;
corrupt_level = 0.4;

lambda = 5;
c_tau = 1.2;

num_trials = 1;

n_near_list = 1:4;
k_list = 1:7;

methods = {'joint','l1_nuc','l1'};

do_plot = false;
verbose = false;

num_methods = length(methods);

%% Storage
heatmap_joint = zeros(length(n_near_list), length(k_list));
heatmap_block = zeros(length(n_near_list), length(k_list));
heatmap_l1 = zeros(length(n_near_list), length(k_list));

%% Sweep
tic;
for ii = 1:length(n_near_list)

    n_near = n_near_list(ii);

    %fprintf('\n=================================\n');
    fprintf('n_near = %d\n', n_near);
    %fprintf('=================================\n');

    for jj = 1:length(k_list)

        k = k_list(jj);

        %fprintf('  k = %d\n', k);

        support_score = zeros(num_trials, num_methods);

        for t = 1:num_trials

            seed = t;

            [Q, A, X_centered, M_tilde, Q_centered, c, corrupted_anchor] = ...
                generate_data(d, n, n_c, sigma, ...
                              n_near, n_far, ...
                              epsilon, k, ...
                              corrupt_level, ...
                              seed, do_plot);

            for r = 1:num_methods

                method = methods{r};

                [S, status] = ...
                    recover_S(M_tilde, X_centered, method, ...
                              n_near, epsilon, lambda, c_tau);

                [bad_rec, Q_hat, ...
                 err_fro, err_mean, err_median, err_max, ...
                 supp_score] = ...
                    clustering_recover_Q( ...
                    S, M_tilde, ...
                    X_centered, Q_centered, c, ...
                    corrupted_anchor, ...
                    n_c, k, verbose);

                support_score(t,r) = supp_score;

            end
        end

        avg_score = mean(support_score,1);

        heatmap_joint(ii,jj) = avg_score(1);
        heatmap_block(ii,jj) = avg_score(2);
        heatmap_l1(ii,jj) = avg_score(3);

    end
end

%% Plot heatmaps

figure;

font_axis = 18;
font_title = 20;
font_tick = 15;

subplot(2,1,1);
imagesc(k_list, n_near_list, heatmap_joint);
colormap(gray);
caxis([0 1]);
colorbar;
set(gca, 'YTick', n_near_list);
set(gca, 'XTick', k_list);
set(gca, 'FontSize', font_tick);
xlabel('Sparsity level k', 'FontSize', font_axis);
ylabel('Near anchors', 'FontSize', font_axis);
title('RSCB', 'FontSize', font_title, 'FontWeight', 'normal');

subplot(2,1,2);
imagesc(k_list, n_near_list, heatmap_block);

colormap(gray);
caxis([0 1]);
colorbar;
set(gca, 'YTick', n_near_list);
set(gca, 'XTick', k_list);
set(gca, 'FontSize', font_tick);
xlabel('Sparsity level k', 'FontSize', font_axis);
ylabel('Near anchors', 'FontSize', font_axis);
title('RSCLNuc', 'FontSize', font_title, 'FontWeight', 'normal');

set(gcf, 'Position', [100, 100, 500, 650]);
set(findall(gcf, 'Type', 'axestoolbar'), 'Visible', 'off');

exportgraphics(gcf, 'heatmap_support_baseline_vs_l1nuc.pdf', 'ContentType', 'vector');
exportgraphics(gcf, 'heatmap_support_baseline_vs_l1nuc.png', 'Resolution', 300);

%% Save data
save('heatmap_support_results.mat', ...
     'heatmap_joint', ...
     'heatmap_block', ...
     'heatmap_l1', ...
     'k_list', ...
     'n_near_list', ...
     'methods', ...
     'num_trials');

total_time = toc;
fprintf('\nTotal runtime = %.2f minutes\n', total_time/60);
fprintf('Done.\n');
