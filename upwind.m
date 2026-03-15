function result = upwind(f, v, dx, BC, BL, BR)
% UPWIND  Compute v . grad_x f using 2nd-order upwind with minmod limiter.
%   result = upwind(f, v, dx, BC, BL, BR)
%
%   Implements a MUSCL-type finite-volume flux for the 1D-2V kinetic
%   transport term  v1 * df/dx  with various boundary conditions.
%
%   Inputs:
%     f   - (Nx x Nv x Nv) distribution function
%     v   - (Nv x 1) velocity nodes
%     dx  - spatial cell width
%     BC  - boundary type: 'specular', 'free flow', 'periodic',
%           'neumann', or 'incoming'
%     BL  - left  boundary data (required for 'incoming' BC)
%     BR  - right boundary data (required for 'incoming' BC)
%
%   Output:
%     result - (Nx x Nv x Nv) approximation of v1 * df/dx

if nargin < 5, BL = []; end
if nargin < 6, BR = []; end

[v1, ~] = ndgrid(v);
Nv = numel(v);
Nx = size(f, 1);
v1 = reshape(v1, 1, Nv, Nv);
result = zeros(size(f));

% Extend with ghost cells
f = applyBC_1D2V(f, BC, BL, BR);

slope = zeros(Nx+3, Nv, Nv);
Flux  = zeros(Nx+1, Nv, Nv);
v1_plus  = (v1 + abs(v1)) / 2;
v1_minus = (v1 - abs(v1)) / 2;

% Minmod slopes
for i = 2:Nx+3
    slope(i,:,:) = minmod(f(i+1,:,:) - f(i,:,:), ...
                          f(i,:,:) - f(i-1,:,:)) / dx;
end

% Reconstructed interface values
F_plus  = f(2:Nx+2,:,:)   + slope(2:Nx+2,:,:) * dx/2;
F_minus = f(3:Nx+3,:,:)   - slope(3:Nx+3,:,:) * dx/2;

% Upwind flux
for i = 1:Nx+1
    Flux(i,:,:) = v1_plus .* F_plus(i,:,:) + v1_minus .* F_minus(i,:,:);
end

% Flux difference
for i = 1:Nx
    result(i,:,:) = (Flux(i+1,:,:) - Flux(i,:,:)) / dx;
end
end


function m = minmod(a, b)
m = (sign(a) + sign(b)) / 2 .* min(abs(a), abs(b));
end


function f_ext = applyBC_1D2V(f, BC, BL, BR)
% Extend f with two ghost cells on each side.
Nx = size(f, 1);
BC = lower(BC);
switch BC
    case 'specular'
        f_ext = cat(1, flip(f(2,:,:),2), flip(f(1,:,:),2), f, ...
                       flip(f(Nx,:,:),2), flip(f(Nx-1,:,:),2));
    case 'free flow'
        f_ext = cat(1, f(1,:,:), f(1,:,:), f, f(Nx,:,:), f(Nx,:,:));
    case 'periodic'
        f_ext = cat(1, f(Nx-1,:,:), f(Nx,:,:), f, f(1,:,:), f(2,:,:));
    case 'neumann'
        f_ext = cat(1, f(2,:,:), f(1,:,:), f, f(Nx,:,:), f(Nx-1,:,:));
    case 'incoming'
        f_ext = cat(1, BL(1,:,:), BL(1,:,:), f, BR(1,:,:), BR(1,:,:));
    otherwise
        error('Invalid boundary condition: %s', BC);
end
end
