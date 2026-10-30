clear; clc;

%% Parameters
d = 2;
n = 99;
n_c = 3;
sigma = 8;

n_near = 2;
n_far = 9;

epsilon = 0.01;
corrupt_level = 0.4;

lambda = 5;
c_tau = 1.2;

num_trials = 10;
k_list = 2:7;
methods = {'joint', 'l1_nuc', 'l1'};

do_plot = false;
verbose = false;

num_methods = length(methods);
num_k = length(k_list);

support_acc = zeros(num_k, num_methods);
median_err = zeros(num_k, num_methods);

%% Sweep over sparsity level k
for kk = 1:num_k
    k = k_list(kk);

    support_score = zeros(num_trials, num_methods);
    median_score = zeros(num_trials, num_methods);

    fprintf('\nRunning k = %d\n', k);

    for t = 1:num_trials
        seed = t;

        [Q, A, X_centered, M_tilde, Q_centered, c, corrupted_anchor] = ...
            generate_data(d, n, n_c, sigma, n_near, n_far, ...
                          epsilon, k, corrupt_level, seed, do_plot);

        for r = 1:num_methods
            method = methods{r};

            [S, status] = recover_S(M_tilde, X_centered, method, ...
                                    n_near, epsilon, lambda, c_tau);

            [bad_rec, Q_hat, err_fro, err_mean, err_median, err_max, supp_score] = ...
                clustering_recover_Q(S, M_tilde, X_centered, Q_centered, c, ...
                                     corrupted_anchor, n_c, k, verbose);

            support_score(t,r) = supp_score;
            median_score(t,r) = err_median;
        end
    end

    support_acc(kk,:) = mean(support_score, 1);
    median_err(kk,:) = mean(median_score, 1);
end

%% Plot 

font_axis = 18; font_tick = 15; font_legend = 13; line_width = 2;

figure;
set(gcf, 'Position', [100, 100, 700, 280]);   
t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% ---- Left: support score ----
nexttile;
set(gca, 'FontSize', font_tick); hold on; grid on;   
plot(k_list, support_acc(:,1), '-o', 'LineWidth', line_width);
plot(k_list, support_acc(:,3), '-^', 'LineWidth', line_width);
plot(k_list, support_acc(:,2), '-s', 'LineWidth', line_width);
xlabel('Sparsity level k', 'FontSize', font_axis);
ylabel('Support score', 'FontSize', font_axis);      
legend('RSCB', 'RSCL', 'RSCLNuc', 'Location', 'southwest', 'FontSize', font_legend);
ylim([0.75, 1.02]); xlim([min(k_list)-0.2, max(k_list)+0.2]); xticks(k_list);

% ---- Right: median error ----
nexttile;
set(gca, 'FontSize', font_tick); hold on; grid on;
plot(k_list, median_err(:,1), '-o', 'LineWidth', line_width);
plot(k_list, median_err(:,3), '-^', 'LineWidth', line_width);
plot(k_list, median_err(:,2), '-s', 'LineWidth', line_width);
xlabel('Sparsity level k', 'FontSize', font_axis);
ylabel('Median relative error', 'FontSize', font_axis);
xlim([min(k_list)-0.2, max(k_list)+0.2]); xticks(k_list);

set(findall(gcf, 'Type', 'axestoolbar'), 'Visible', 'off');

exportgraphics(gcf, 'curves.pdf', 'ContentType', 'vector');
exportgraphics(gcf, 'curves.png', 'Resolution', 300);

%% Save results
save('curves_vs_k_results.mat', ...
     'support_acc', 'median_err', ...
     'k_list', 'methods', ...
     'n_near', 'n_far', 'epsilon', ...
     'corrupt_level', 'num_trials');
