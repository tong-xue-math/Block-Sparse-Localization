function [S, status] = recover_S(M_tilde, X_centered, method, ...
                                 n_near, epsilon, lambda, c_tau)

[m, n] = size(M_tilde);
d = size(X_centered, 2);

switch method

    case 'joint'
        Q_joint = zeros(d, n);
        S = zeros(m, n);
        status = cell(n, 1);

        B = [X_centered, eye(m)];

        for ell = 1:n
            mtilde = M_tilde(:, ell);

            cvx_begin quiet
                variable u(d + m, 1)
                minimize( norm(u(d+1:end), 1) )
                subject to
                    norm(mtilde - B * u, 'fro') <= tau_R;
            cvx_end

            Q_joint(:, ell) = u(1:d);
            S(:, ell) = u(d+1:end);
            status{ell} = cvx_status;
        end

    case 'l1_nuc'
        R = null(X_centered')';

        sigma_E = epsilon * norm(M_tilde, 'fro') / sqrt(numel(M_tilde));
        tau_R = c_tau * sigma_E * sqrt(size(R,1) * n);

        cvx_begin quiet
            variable S(m,n)
            minimize( norm(S(:), 1) + lambda * norm_nuc(S) )
            subject to
                norm(R * (M_tilde - S), 'fro') <= tau_R;
                S(1:n_near, :) == 0;
        cvx_end

        status = cvx_status;

    case 'l1'
        R = null(X_centered')';

        sigma_E = epsilon * norm(M_tilde, 'fro') / sqrt(numel(M_tilde));
        tau_R = c_tau * sigma_E * sqrt(size(R,1) * n);

        cvx_begin quiet
            variable S(m,n)
            minimize( norm(S(:), 1) )
            subject to
                norm(R * (M_tilde - S), 'fro') <= tau_R;
                S(1:n_near, :) == 0;
        cvx_end

        status = cvx_status;

    otherwise
        error('Unknown recovery method.');
end

end
