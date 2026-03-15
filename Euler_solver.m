function [rho, u, theta, snapshots] = Euler_solver(f0, v, dx, dt, nsteps, ...
        Nx, Nv, time_slices, RK_type, ML, MR)
% EULER_SOLVER  Euler-limit solver (transport + Maxwellian projection).
%   [rho, u, theta, snapshots] = Euler_solver(f0, v, dx, dt, nsteps, ...
%       Nx, Nv, time_slices, RK_type, ML, MR)
%
%   Solves the compressible Euler equations by repeatedly transporting the
%   distribution function and projecting onto the local Maxwellian.  Used to
%   generate reference solutions in the hydrodynamic limit (epsilon -> 0).
%
%   Inputs:
%     f0          - (Nx x Nv x Nv) initial distribution
%     v           - (Nv x 1) velocity nodes
%     dx, dt      - spatial / temporal step sizes
%     nsteps      - total number of time steps
%     Nx, Nv      - grid dimensions
%     time_slices - target snapshot times
%     RK_type     - 'first_order' or 'second_order'
%     ML, MR      - boundary Maxwellians (optional)
%
%   Outputs:
%     rho, u, theta - final macroscopic quantities
%     snapshots     - containers.Map of snapshot structs

if nargin < 11, ML = []; MR = []; end
if nargin < 9, RK_type = 'first_order'; end

F = f0;
t = 0;

snapshots = containers.Map('KeyType', 'double', 'ValueType', 'any');
closest_diff = inf(length(time_slices), 1);
closest_data = cell(length(time_slices), 1);
report_every = max(1, floor(nsteps / 100));

for n = 1:nsteps
    if strcmp(RK_type, 'first_order')
        F1 = F - dt .* upwind(F, v, dx, 'neumann', ML, MR);
        rho = trapz(v, trapz(v, F1, 2), 3);
        u = zeros(Nx, 2);
        u(:,1) = trapz(v, trapz(v, reshape(v,1,Nv,1).*F1, 2), 3) ./ rho;
        u(:,2) = trapz(v, trapz(v, reshape(v,1,1,Nv).*F1, 2), 3) ./ rho;
        theta = trapz(v, trapz(v, ((u(:,1)-v.').^2 + ...
            reshape((u(:,2)-v.').^2,Nx,1,Nv)).*F1, 2), 3) ./ rho ./ 2;
        F = Maxwellian(rho, u, theta, v);
        t = t + dt;

    elseif strcmp(RK_type, 'second_order')
        % ── Stage 1 (half step) ──
        tr_mid = F - dt/2 .* upwind(F, v, dx, 'neumann', ML, MR);
        rho_m = trapz(v, trapz(v, tr_mid, 2), 3);
        u_m = zeros(Nx, 2);
        u_m(:,1) = trapz(v, trapz(v, reshape(v,1,Nv,1).*tr_mid, 2), 3) ./ rho_m;
        u_m(:,2) = trapz(v, trapz(v, reshape(v,1,1,Nv).*tr_mid, 2), 3) ./ rho_m;
        theta_m = trapz(v, trapz(v, ((u_m(:,1)-v.').^2 + ...
            reshape((u_m(:,2)-v.').^2,Nx,1,Nv)).*tr_mid, 2), 3) ./ rho_m ./ 2;
        F_mid = Maxwellian(rho_m, u_m, theta_m, v);

        % ── Stage 2 (full step) ──
        tr_new = F - dt .* upwind(F_mid, v, dx, 'neumann', ML, MR);
        rho_n = trapz(v, trapz(v, tr_new, 2), 3);
        u_n = zeros(Nx, 2);
        u_n(:,1) = trapz(v, trapz(v, reshape(v,1,Nv,1).*tr_new, 2), 3) ./ rho_n;
        u_n(:,2) = trapz(v, trapz(v, reshape(v,1,1,Nv).*tr_new, 2), 3) ./ rho_n;
        theta_n = trapz(v, trapz(v, ((u_n(:,1)-v.').^2 + ...
            reshape((u_n(:,2)-v.').^2,Nx,1,Nv)).*tr_new, 2), 3) ./ rho_n ./ 2;
        F = Maxwellian(rho_n, u_n, theta_n, v);
        t = t + dt;

    else
        error('RK_type must be ''first_order'' or ''second_order''');
    end

    % Record closest snapshot for each target time
    for i = 1:length(time_slices)
        tdiff = abs(t - time_slices(i));
        if tdiff < closest_diff(i)
            closest_diff(i) = tdiff;
            snap.rho   = trapz(v, trapz(v, F, 2), 3);
            u_s = zeros(Nx, 2);
            u_s(:,1) = trapz(v, trapz(v, reshape(v,1,Nv,1).*F, 2), 3) ./ snap.rho;
            u_s(:,2) = trapz(v, trapz(v, reshape(v,1,1,Nv).*F, 2), 3) ./ snap.rho;
            snap.u1    = u_s(:,1);
            snap.theta = trapz(v, trapz(v, ((u_s(:,1)-v.').^2 + ...
                reshape((u_s(:,2)-v.').^2,Nx,1,Nv)).*F, 2), 3) ./ snap.rho ./ 2;
            closest_data{i} = snap;
        end
    end

    if mod(n, report_every) == 0 || n == 1 || n == nsteps
        fprintf('Euler solver: %.1f%%\n', n/nsteps*100);
    end
end

for i = 1:length(time_slices)
    if ~isempty(closest_data{i})
        snapshots(time_slices(i)) = closest_data{i};
    end
end

rho = trapz(v, trapz(v, F, 2), 3);
u = zeros(Nx, 2);
u(:,1) = trapz(v, trapz(v, reshape(v,1,Nv,1).*F, 2), 3) ./ rho;
u(:,2) = trapz(v, trapz(v, reshape(v,1,1,Nv).*F, 2), 3) ./ rho;
theta = trapz(v, trapz(v, ((u(:,1)-v.').^2 + ...
    reshape((u(:,2)-v.').^2,Nx,1,Nv)).*F, 2), 3) ./ rho ./ 2;
end
