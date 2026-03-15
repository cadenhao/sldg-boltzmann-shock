function result = Maxwellian(rho, u, theta, v)
% MAXWELLIAN  Evaluate the 2-D Maxwellian distribution.
%   M = Maxwellian(rho, u, theta, v)
%
%   Inputs:
%     rho   - (Nx x 1) density
%     u     - (Nx x 2) bulk velocity [u1, u2]
%     theta - (Nx x 1) temperature
%     v     - (Nv x 1) velocity nodes in each direction
%
%   Output:
%     result - (Nx x Nv x Nv) Maxwellian distribution
%
%   M = rho / (2 pi theta) * exp( -[ (u1-v1)^2 + (u2-v2)^2 ] / (2 theta) )

Nx = size(rho, 1);
Nv = size(v, 1);
result = rho ./ (2*pi*theta) .* exp( ...
    -( (u(:,1) - v.').^2 + reshape((u(:,2) - v.').^2, Nx, 1, Nv) ) ...
    ./ (2*theta) );
end
