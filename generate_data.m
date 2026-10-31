function [Q, A, X_centered, M_tilde, Q_centered, c, corrupted_anchor, S_true_support] = ...
    generate_data(d, n, n_c, sigma, n_near, n_far, ...
                  epsilon, k, corrupt_level, seed, do_plot)

rng(seed);

m = n_near + n_far;

% Target and near anchor region
near_x_range = [-40, 40];
near_y_range = [-40, 40];

% Far anchor regions
far_left_x_range  = [-140, -100];
far_right_x_range = [100, 140];
far_y_range       = [-60, 40];

%% Generate target nodes
p_c = n / n_c;

centers = [
    -30,  30,   0;
    -15, -15,  27
];

Q = [];

for i = 1:n_c
    cluster_i = centers(:, i) + sigma * randn(d, p_c);
    Q = [Q, cluster_i];
end

%% Generate anchors
A = zeros(d, m);

% Near anchors
for j = 1:n_near
    x = near_x_range(1) + diff(near_x_range) * rand;
    y = near_y_range(1) + diff(near_y_range) * rand;
    A(:, j) = [x; y];
end

% Far anchors
for j = 1:n_far
    anchor_idx = n_near + j;

    if j <= ceil(n_far / 2)
        x_range = far_left_x_range;
    else
        x_range = far_right_x_range;
    end

    x = x_range(1) + diff(x_range) * rand;
    y = far_y_range(1) + diff(far_y_range) * rand;

    A(:, anchor_idx) = [x; y];
end

%% Optional plot
if do_plot
    figure;

    scatter(Q(1,:), Q(2,:), 40, 'filled');
    hold on;

    scatter(centers(1,:), centers(2,:), 120, 'x', 'LineWidth', 2);
    scatter(A(1,1:n_near), A(2,1:n_near), 100, '^', 'filled');
    scatter(A(1,n_near+1:end), A(2,n_near+1:end), 100, 's', 'filled');

    for i = 1:n_c
        text(centers(1,i), centers(2,i), sprintf('  C%d', i), ...
            'FontSize', 12, 'FontWeight', 'bold');
    end

    for j = 1:m
        text(A(1,j), A(2,j), sprintf('  A%d', j), ...
            'FontSize', 12, 'FontWeight', 'bold');
    end

    xlim([-150, 150]);
    ylim([-60, 60]);
    grid on;
    xlabel('x');
    ylabel('y');
    title(sprintf('%d targets in %d clusters, %d near anchors, and %d far anchors', ...
        n, n_c, n_near, n_far));
    legend('Targets', 'Cluster Centers', 'Near Anchors', 'Far Anchors');
end

%% Generate distances
D_true = zeros(n, m);

for j = 1:m
    D_true(:, j) = vecnorm(Q - A(:, j), 2, 1)';
end

Dsq_true = D_true.^2;

Z = randn(n, m);
Dsq_noisy = Dsq_true .* (1 + epsilon * Z);

%% Choose corrupted far anchors
far_anchor_idx = n_near + 1 : m;
corrupted_anchor = cell(n_c, 1);

for i = 1:n_c
    corrupted_anchor{i} = far_anchor_idx(randperm(n_far, k));
end

%% Add sparse corruptions
S_D = zeros(n, m);

for i = 1:n_c
    target_idx_i = (i - 1) * p_c + 1 : i * p_c;
    bad_i = corrupted_anchor{i};

    signs = sign(randn(length(target_idx_i), length(bad_i)));

    S_D(target_idx_i, bad_i) = ...
        corrupt_level * Dsq_true(target_idx_i, bad_i) .* signs;
end

S_true_support = (S_D ~= 0)';   % m x n

Dsq_tilde = Dsq_noisy + S_D;

%% Construct M_tilde
all_corrupted = unique([corrupted_anchor{:}]);
candidates = setdiff(1:m, all_corrupted, 'stable');
c = candidates(1);

A_centered = A - A(:, c);
Q_centered = Q - A(:, c);

X_centered = A_centered';

M_tilde = zeros(m, n);

for j = 1:m
    for i = 1:n
        M_tilde(j,i) = 0.5 * ...
            (Dsq_tilde(i,c) - Dsq_tilde(i,j) + norm(A_centered(:,j))^2);
    end
end

end
