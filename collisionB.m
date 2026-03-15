function [Q, Q_minus] = collisionB(g, h, Nx, Nv, Mv, Rv, Lv)
% COLLISIONB  Spectral method for the 2-D Boltzmann collision operator.
%   [Q, Q_minus] = collisionB(g, h, Nx, Nv, Mv, Rv, Lv)
%
%   Implements the Fourier-spectral approach of Mouhot & Pareschi (2006)
%   for Maxwell molecules on a uniform velocity grid.
%
%   Inputs:
%     g, h - (Nx x Nv x Nv) distribution functions
%     Nx   - number of spatial points
%     Nv   - number of velocity modes per direction
%     Mv   - number of angular quadrature points
%     Rv   - support radius for the kernel
%     Lv   - velocity domain half-width
%
%   Outputs:
%     Q       - (Nx x Nv x Nv) collision integral Q(g, h)
%     Q_minus - (Nx x Nv x Nv) loss part of the collision operator

% Fourier frequencies
[l1, l2] = ndgrid([0:Nv/2-1, -Nv/2:-1]);
l1 = pi / Lv * l1;
l2 = pi / Lv * l2;

Q = zeros(Nx, Nv, Nv);
Q_minus = zeros(Nx, Nv, Nv);

% Angular quadrature on [0, pi)
theta = reshape(linspace(0.5*pi/Mv, pi - 0.5*pi/Mv, Mv), 1, 1, Mv);

% Kernel modes
proj = l1 .* cos(theta) + l2 .* sin(theta);
alpha  = sqrt(2/(2*pi)) * 2*Rv .* mysinc(Rv * proj);
alpha_ = sqrt(2/(2*pi)) * 2*Rv .* mysinc(Rv * sqrt(l1.^2 + l2.^2 - proj.^2));

for i = 1:Nx
    gi = squeeze(g(i,:,:));
    hi = squeeze(h(i,:,:));
    Q(i,:,:) = pi/Mv * real(sum( ...
        ifft2(alpha .* fft2(gi)) .* ifft2(alpha_ .* fft2(hi)) ...
        - gi .* ifft2(fft2(hi) .* alpha .* alpha_), 3));
    Q_minus(i,:,:) = pi/Mv * real(sum( ...
        ifft2(fft2(hi) .* alpha .* alpha_), 3));
end
end


function y = mysinc(x)
% Un-normalised sinc:  sin(x) / x,  with sinc(0) = 1.
y = ones(size(x));
nz = (x ~= 0);
y(nz) = sin(x(nz)) ./ x(nz);
end
