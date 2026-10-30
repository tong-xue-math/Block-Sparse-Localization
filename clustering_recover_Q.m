function [bad_rec, Q_hat, err_fro, err_mean, err_median, err_max, supp_score, S_support] = ...
    clustering_recover_Q(S, M_tilde, X_centered, Q_centered, c, ...
                         corrupted_anchor, n_c, k, verbose)

[m, n] = size(S);
d = size(X_centered, 2);

%% Keep the top-k entries in each column
S_top = zeros(size(S));
keep_idx_all = zeros(k, n);

for j = 1:n
    [~, idx] = sort(abs(S(:, j)), 'descend');
    keep_idx = idx(1:k);

    keep_idx_all(:, j) = keep_idx;
    S_top(keep_idx, j) = S(keep_idx, j);
end

%% Graph Laplacian clustering based on S_top
W = zeros(n, n);

for i = 1:n
    supp_i = find(S_top(:, i) ~= 0);

    for j = i+1:n
        supp_j = find(S_top(:, j) ~= 0);

        shared = numel(intersect(supp_i, supp_j));
        union_size = numel(union(supp_i, supp_j));

        if union_size > 0
            W(i, j) = shared / union_size;
            W(j, i) = W(i, j);
        end
    end
end

degree = sum(W, 2);
D_inv_sqrt = diag(1 ./ sqrt(degree + eps));
L_norm = eye(n) - D_inv_sqrt * W * D_inv_sqrt;

[V, Eval] = eig(L_norm);
[~, order] = sort(diag(Eval), 'ascend');
U = V(:, order(1:n_c));

U = U ./ (vecnorm(U, 2, 2) + eps);

cluster_label = kmeans(U, n_c, 'Replicates', 20);

%% Recover corrupted anchors for each recovered cluster
recovered_anchor = cell(n_c, 1);
S_support = false(m, n);

for i = 1:n_c
    target_idx_i = find(cluster_label == i);

    anchors_i_all = keep_idx_all(:, target_idx_i);

    counts = histcounts(anchors_i_all(:), 1:m+1);

    [~, idx] = sort(counts, 'descend');
    anchors_i = idx(1:k);

    S_support(anchors_i, target_idx_i) = true;

    recovered_anchor{i} = anchors_i;
end

%% Match recovered clusters with true clusters
overlap = zeros(n_c, n_c);

for i = 1:n_c
    for j = 1:n_c
        overlap(i,j) = numel(intersect(corrupted_anchor{i}, recovered_anchor{j}));
    end
end

P = perms(1:n_c);
score = zeros(size(P,1), 1);

for r = 1:size(P,1)
    for i = 1:n_c
        score(r) = score(r) + overlap(i, P(r,i));
    end
end

[best_score, best_idx] = max(score);
best_match = P(best_idx, :);

bad_rec = cell(n_c, 1);

for i = 1:n_c
    rec_i = best_match(i);
    bad_rec{i} = recovered_anchor{rec_i};
end

supp_score = best_score / (n_c * k);

if verbose
    fprintf('Recovered support indicates:\n');
    for i = 1:n_c
        fprintf('Recovered cluster %d has corrupted anchors: ', i);
        fprintf('A%d ', bad_rec{i});
        fprintf('\n');
    end
end

%% Recover Q by least squares
Q_hat = zeros(d, n);

for j = 1:n
    good_idx = ~S_support(:, j);

    % The reference-anchor row has no information after centering
    good_idx(c) = false;

    X_good = X_centered(good_idx, :);
    M_good = M_tilde(good_idx, j);

    Q_hat(:, j) = X_good \ M_good;
end

%% Compute errors
err_fro = norm(Q_hat - Q_centered, 'fro') / norm(Q_centered, 'fro');

err_per_node = vecnorm(Q_hat - Q_centered, 2, 1);
rerr_per_node = err_per_node ./ max(vecnorm(Q_centered, 2, 1), 1e-12);

err_mean = mean(rerr_per_node);
err_median = median(rerr_per_node);
err_max = max(rerr_per_node);

end