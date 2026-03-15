function [rho, u, theta, snapshots] = FD_solver(f0, v, dx, dt, nsteps, ...
        epsilon, Nx, Mv, Rv, Lv, Nv, time_slices, RK_type, ML, MR)
% FD_SOLVER  Finite-difference IMEX solver for the Boltzmann equation.
%   [rho, u, theta, snapshots] = FD_solver(f0, v, dx, dt, nsteps, ...
%       epsilon, Nx, Mv, Rv, Lv, Nv, time_slices, RK_type, ML, MR)
%
%   Solves the 1D-2V Boltzmann equation on the Sod shock tube domain using
%   a finite-difference discretisation in space (upwind + minmod) with IMEX
%   time stepping (BGK penalisation).  This solver serves as a reference
%   solution for comparison against the SLDG-IMEX Python implementation.
%
%   Inputs:
%     f0          - (Nx x Nv x Nv) initial distribution function
%     v           - (Nv x 1) velocity nodes
%     dx          - spatial cell width
%     dt          - time step
%     nsteps      - total number of time steps
%     epsilon     - Knudsen number
%     Nx          - number of spatial cells
%     Mv, Rv, Lv  - spectral collision parameters
%     Nv          - number of velocity nodes per direction
%     time_slices - target times for snapshots
%     RK_type     - 'first_order' or 'second_order'
%     ML, MR      - boundary Maxwellians (optional)
%
%   Outputs:
%     rho, u, theta - final macroscopic quantities
%     snapshots     - containers.Map of snapshot structs keyed by time

if nargin < 15, ML = []; MR = []; end
if nargin < 13, RK_type = 'first_order'; end

f = f0;
t = 0;

snapshots = containers.Map('KeyType', 'double', 'ValueType', 'any');
closest_diff = inf(length(time_slices), 1);
closest_data = cell(length(time_slices), 1);
report_every = max(1, floor(nsteps / 100));

for n = 1:nsteps
    if strcmp(RK_type, 'first_order')
        % ── Stage 1 ──
        rho = trapz(v, trapz(v, f, 2), 3);
        u = zeros(Nx, 2);
        u(:,1) = trapz(v, trapz(v, reshape(v,1,Nv,1).*f, 2), 3) ./ rho;
        u(:,2) = trapz(v, trapz(v, reshape(v,1,1,Nv).*f, 2), 3) ./ rho;
        theta = trapz(v, trapz(v, ((u(:,1)-v.').^2 + ...
            reshape((u(:,2)-v.').^2,Nx,1,Nv)).*f, 2), 3) ./ rho ./ 2;
        M = Maxwellian(rho, u, theta, v);

        transport = f - dt .* upwind(f, v, dx, 'neumann', ML, MR);
        rho1 = trapz(v, trapz(v, transport, 2), 3);
        u1 = zeros(Nx, 2);
        u1(:,1) = trapz(v, trapz(v, reshape(v,1,Nv,1).*transport, 2), 3) ./ rho1;
        u1(:,2) = trapz(v, trapz(v, reshape(v,1,1,Nv).*transport, 2), 3) ./ rho1;
        theta1 = trapz(v, trapz(v, ((u1(:,1)-v.').^2 + ...
            reshape((u1(:,2)-v.').^2,Nx,1,Nv)).*transport, 2), 3) ./ rho1 ./ 2;
        M1 = Maxwellian(rho1, u1, theta1, v);

        Q = collisionB(f, f, Nx, Nv, Mv, Rv, Lv);
        beta = rho;
        beta1 = rho1;
        f = epsilon./(epsilon + beta1.*dt) .* transport ...
            + dt./(epsilon + beta1.*dt) .* (Q - beta.*(M - f)) ...
            + beta1.*dt./(epsilon + beta1.*dt) .* M1;
        t = t + dt;

    elseif strcmp(RK_type, 'second_order')
        % ── Stage 1 (half step) ──
        rho = trapz(v, trapz(v, f, 2), 3);
        u = zeros(Nx, 2);
        u(:,1) = trapz(v, trapz(v, reshape(v,1,Nv,1).*f, 2), 3) ./ rho;
        u(:,2) = trapz(v, trapz(v, reshape(v,1,1,Nv).*f, 2), 3) ./ rho;
        theta = trapz(v, trapz(v, ((u(:,1)-v.').^2 + ...
            reshape((u(:,2)-v.').^2,Nx,1,Nv)).*f, 2), 3) ./ rho ./ 2;
        M = Maxwellian(rho, u, theta, v);
        beta = rho;

        tr_mid = f - dt/2 .* upwind(f, v, dx, 'neumann', ML, MR);
        rho_m = trapz(v, trapz(v, tr_mid, 2), 3);
        u_m = zeros(Nx, 2);
        u_m(:,1) = trapz(v, trapz(v, reshape(v,1,Nv,1).*tr_mid, 2), 3) ./ rho_m;
        u_m(:,2) = trapz(v, trapz(v, reshape(v,1,1,Nv).*tr_mid, 2), 3) ./ rho_m;
        theta_m = trapz(v, trapz(v, ((u_m(:,1)-v.').^2 + ...
            reshape((u_m(:,2)-v.').^2,Nx,1,Nv)).*tr_mid, 2), 3) ./ rho_m ./ 2;
        M_m = Maxwellian(rho_m, u_m, theta_m, v);
        beta_m = rho_m;

        Q = collisionB(f, f, Nx, Nv, Mv, Rv, Lv);
        f_mid = epsilon./(epsilon + beta_m.*dt/2) .* tr_mid ...
            + dt/2./(epsilon + beta_m.*dt/2) .* (Q - beta.*(M - f)) ...
            + beta_m.*dt/2./(epsilon + beta_m.*dt/2) .* M_m;

        % ── Stage 2 (full step) ──
        rho_m = trapz(v, trapz(v, f_mid, 2), 3);
        u_m = zeros(Nx, 2);
        u_m(:,1) = trapz(v, trapz(v, reshape(v,1,Nv,1).*f_mid, 2), 3) ./ rho_m;
        u_m(:,2) = trapz(v, trapz(v, reshape(v,1,1,Nv).*f_mid, 2), 3) ./ rho_m;
        theta_m = trapz(v, trapz(v, ((u_m(:,1)-v.').^2 + ...
            reshape((u_m(:,2)-v.').^2,Nx,1,Nv)).*f_mid, 2), 3) ./ rho_m ./ 2;
        M_m = Maxwellian(rho_m, u_m, theta_m, v);
        beta_m = rho_m;

        tr_new = f - dt .* upwind(f_mid, v, dx, 'neumann', ML, MR);
        rho_n = trapz(v, trapz(v, tr_new, 2), 3);
        u_n = zeros(Nx, 2);
        u_n(:,1) = trapz(v, trapz(v, reshape(v,1,Nv,1).*tr_new, 2), 3) ./ rho_n;
        u_n(:,2) = trapz(v, trapz(v, reshape(v,1,1,Nv).*tr_new, 2), 3) ./ rho_n;
        theta_n = trapz(v, trapz(v, ((u_n(:,1)-v.').^2 + ...
            reshape((u_n(:,2)-v.').^2,Nx,1,Nv)).*tr_new, 2), 3) ./ rho_n ./ 2;
        M_n = Maxwellian(rho_n, u_n, theta_n, v);
        beta_n = rho_n;

        Q_mid = collisionB(f_mid, f_mid, Nx, Nv, Mv, Rv, Lv);
        f = epsilon./(epsilon + beta_n.*dt/2) .* tr_new ...
            + dt./(epsilon + beta_n.*dt/2) .* (Q_mid - beta_m.*(M_m - f_mid)) ...
            + beta_n.*dt./(2*epsilon + beta_n.*dt) .* M_n ...
            + beta_m.*dt./(2*epsilon + beta_n.*dt) .* (M_m - f_mid);
        t = t + dt;

    else
        error('RK_type must be ''first_order'' or ''second_order''');
    end

    % Record closest snapshot for each target time
    for i = 1:length(time_slices)
        tdiff = abs(t - time_slices(i));
        if tdiff < closest_diff(i)
            closest_diff(i) = tdiff;
            snap.rho   = trapz(v, trapz(v, f, 2), 3);
            u_s = zeros(Nx, 2);
            u_s(:,1) = trapz(v, trapz(v, reshape(v,1,Nv,1).*f, 2), 3) ./ snap.rho;
            u_s(:,2) = trapz(v, trapz(v, reshape(v,1,1,Nv).*f, 2), 3) ./ snap.rho;
            snap.u1    = u_s(:,1);
            snap.theta = trapz(v, trapz(v, ((u_s(:,1)-v.').^2 + ...
                reshape((u_s(:,2)-v.').^2,Nx,1,Nv)).*f, 2), 3) ./ snap.rho ./ 2;
            closest_data{i} = snap;
        end
    end

    if mod(n, report_every) == 0 || n == 1 || n == nsteps
        fprintf('FD solver: %.1f%%\n', n/nsteps*100);
    end
end

% Store final snapshots
for i = 1:length(time_slices)
    if ~isempty(closest_data{i})
        snapshots(time_slices(i)) = closest_data{i};
    end
end

% Final moments
rho = trapz(v, trapz(v, f, 2), 3);
u = zeros(Nx, 2);
u(:,1) = trapz(v, trapz(v, reshape(v,1,Nv,1).*f, 2), 3) ./ rho;
u(:,2) = trapz(v, trapz(v, reshape(v,1,1,Nv).*f, 2), 3) ./ rho;
theta = trapz(v, trapz(v, ((u(:,1)-v.').^2 + ...
    reshape((u(:,2)-v.').^2,Nx,1,Nv)).*f, 2), 3) ./ rho ./ 2;
end
