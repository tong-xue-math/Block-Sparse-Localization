clear; clc;

%% Parameters
d = 2;
n = 99;
n_c = 3;
sigma = 8;

n_near = 2;
n_far = 9;

epsilon = 0.01;
k = 5;
corrupt_level = 0.4;

lambda = 5;
c_tau = 1.2;

num_trials = 50;
methods = {'joint', 'l1_nuc', 'l1'};

do_plot = false;
verbose = false;

%% Storage
num_methods = length(methods);

support_score = zeros(num_trials, num_methods);
q_fro_err = zeros(num_trials, num_methods);
q_mean_err = zeros(num_trials, num_methods);
q_median_err = zeros(num_trials, num_methods);
q_max_err = zeros(num_trials, num_methods);
solved = false(num_trials, num_methods);

results = cell(num_trials, num_methods);
data_store = cell(num_trials, 1);

%% Main loop
for t = 1:num_trials
    seed = t;

    fprintf('\nTrial %d/%d\n', t, num_trials);

    [Q, A, X_centered, M_tilde, Q_centered, c, corrupted_anchor] = ...
        generate_data(d, n, n_c, sigma, n_near, n_far, ...
                      epsilon, k, corrupt_level, seed, do_plot);

    data_store{t}.seed = seed;
    data_store{t}.Q = Q;
    data_store{t}.A = A;
    data_store{t}.X_centered = X_centered;
    data_store{t}.M_tilde = M_tilde;
    data_store{t}.Q_centered = Q_centered;
    data_store{t}.c = c;
    data_store{t}.corrupted_anchor = corrupted_anchor;

    for r = 1:num_methods
        method = methods{r};

        fprintf('  Method: %s\n', method);

        [S, status] = recover_S(M_tilde, X_centered, method, ...
                                n_near, epsilon, lambda, c_tau);

        [bad_rec, Q_hat, err_fro, err_mean, err_median, err_max, supp_score] = ...
            clustering_recover_Q(S, M_tilde, X_centered, Q_centered, c, ...
                                 corrupted_anchor, n_c, k, verbose);

        support_score(t,r) = supp_score;
        q_fro_err(t,r) = err_fro;
        q_mean_err(t,r) = err_mean;
        q_median_err(t,r) = err_median;
        q_max_err(t,r) = err_max;

        if iscell(status)
            solved(t,r) = all(strcmp(status, 'Solved') | strcmp(status, 'Inaccurate/Solved'));
        else
            solved(t,r) = strcmp(status, 'Solved') || strcmp(status, 'Inaccurate/Solved');
        end

        results{t,r}.method = method;
        results{t,r}.seed = seed;
        results{t,r}.status = status;
        results{t,r}.S = S;
        results{t,r}.bad_rec = bad_rec;
        results{t,r}.Q_hat = Q_hat;
        results{t,r}.support_acc = supp_score;
        results{t,r}.err_fro = err_fro;
        results{t,r}.err_mean = err_mean;
        results{t,r}.err_median = err_median;
        results{t,r}.err_max = err_max;
        results{t,r}.solved = solved(t,r);
    end
end

%% Average results
avg_solved = mean(solved, 1);
avg_support_score = mean(support_score, 1);
avg_q_fro_err = mean(q_fro_err, 1);
avg_q_mean_err = mean(q_mean_err, 1);
avg_q_median_err = mean(q_median_err, 1);
avg_q_max_err = mean(q_max_err, 1);

%% Standard deviation results
sd_solved = std(double(solved), 0, 1);
sd_support_score = std(support_score, 0, 1);
sd_q_fro_err = std(q_fro_err, 0, 1);
sd_q_mean_err = std(q_mean_err, 0, 1);
sd_q_median_err = std(q_median_err, 0, 1);
sd_q_max_err = std(q_max_err, 0, 1);

%% Print average results
fprintf('\n==============================\n');
fprintf('Average over %d trials\n', num_trials);
fprintf('==============================\n');

for r = 1:num_methods
    fprintf('\nMethod: %s\n', methods{r});
    fprintf('Solve rate: %.2f ± %.2f\n', avg_solved(r), sd_solved(r));
    fprintf('Support accuracy: %.4f ± %.4f\n', avg_support_score(r), sd_support_score(r));
    fprintf('Q Fro relative error: %.4e ± %.4e\n', avg_q_fro_err(r), sd_q_fro_err(r));
    fprintf('Mean node error: %.4e ± %.4e\n', avg_q_mean_err(r), sd_q_mean_err(r));
    fprintf('Median node error: %.4e ± %.4e\n', avg_q_median_err(r), sd_q_median_err(r));
    fprintf('Max node error: %.4e ± %.4e\n', avg_q_max_err(r), sd_q_max_err(r));
end

%% Save results
save('compare_recovery_results.mat', ...
     'results', 'data_store', 'methods', ...
     'support_score', 'q_fro_err', 'q_mean_err', ...
     'q_median_err', 'q_max_err', 'solved', ...
     'avg_solved', 'avg_support_score', ...
     'avg_q_fro_err', 'avg_q_mean_err', ...
     'avg_q_median_err', 'avg_q_max_err', ...
     'sd_solved', 'sd_support_score', ...
     'sd_q_fro_err', 'sd_q_mean_err', ...
     'sd_q_median_err', 'sd_q_max_err');