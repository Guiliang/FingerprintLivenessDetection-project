function [h1d Excitation Orientation h2d] = WLD_new(image,radius,neighbors,Ebin,Obin,Sbin)

% ==========================Parameters Setting=========================
alpha = 3;
beta = 5;
EPSILON = 0.0000001;
if nargin<4
    Ebin = 6;
    Obin = 8;
    Sbin = 20;
end
%======================================================================

%% ====================Differential Excitation================

[ysize xsize] = size(image);
image = double(image);

if nargin<2
    spoints=[-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];
else
    spoints = zeros(neighbors,2);

    % Angle step.
    a = 2*pi/neighbors;
    
    for i = 1:neighbors
        spoints(i,1) = -radius*sin((i-1)*a);
        spoints(i,2) = radius*cos((i-1)*a);
    end
end

neighbors = size(spoints,1);

miny = min(spoints(:,1));
maxy = max(spoints(:,1));
minx = min(spoints(:,2));
maxx = max(spoints(:,2));

% Block size, each LBP code is computed within a block of size bsizey*bsizex
bsizey = ceil(max(maxy,0))-floor(min(miny,0))+1;
bsizex = ceil(max(maxx,0))-floor(min(minx,0))+1;

% Coordinates of origin (0,0) in the block
origy = 1-floor(min(miny,0));
origx = 1-floor(min(minx,0));

% Minimum allowed size for the input image depends
% on the radius of the used LBP operator.
if(xsize < bsizex || ysize < bsizey)
  error('Too small input image. Should be at least (2*radius+1) x (2*radius+1)');
end

% Calculate dx and dy;
dx = xsize - bsizex;
dy = ysize - bsizey;

% Fill the center pixel matrix C.
Cen = image(origy:origy+dy,origx:origx+dx);

% Initialize the result matrix with zeros.
result = zeros(dy+1,dx+1);

for i = 1:neighbors
    y = spoints(i,1)+origy;
    x = spoints(i,2)+origx;
    % Calculate floors, ceils and rounds for the x and y.
    fy = floor(y); cy = ceil(y); ry = round(y);
    fx = floor(x); cx = ceil(x); rx = round(x);
    % Check if interpolation is needed.
    if (abs(x - rx) < 1e-6) && (abs(y - ry) < 1e-6)
        % Interpolation is not needed, use original datatypes
        N = image(ry:ry+dy,rx:rx+dx);
        D = N-Cen;
    else
        % Interpolation needed, use double type images 
        ty = y - fy;
        tx = x - fx;
        % Calculate the interpolation weights.
        w1 = (1 - tx) * (1 - ty);
        w2 =      tx  * (1 - ty);
        w3 = (1 - tx) *      ty ;
        w4 =      tx  *      ty ;
        % Compute interpolated pixel values
        N = w1*image(fy:fy+dy,fx:fx+dx) + w2*image(fy:fy+dy,cx:cx+dx) + ...
            w3*image(cy:cy+dy,fx:fx+dx) + w4*image(cy:cy+dy,cx:cx+dx);
        D = N-Cen; 
    end 
    % Update the result matrix.    
    result = result + D;
end

Excitation = atan(alpha*result./(Cen+beta));
Excitation(Cen==0) = 0.1;
%========================================================================

%% =========================Orientation=========================
V10 = image(origy+1:origy+dy+1,origx:origx+dx)-image(origy-1:origy+dy-1,origx:origx+dx);
V11 = image(origy:origy+dy,origx-1:origx+dx-1)-image(origy:origy+dy,origx+1:origx+dx+1);

theta = atan(V10./V11);
theta(abs(V11)<EPSILON) = 0;

theta = theta*180/pi;
theta2 = zeros(size(theta));

set1 = find(V11 > EPSILON & V10 > EPSILON);
set2 = find(V11 < -EPSILON & V10 > EPSILON);
set3 = find(V11 < -EPSILON & V10 < -EPSILON);
set4 = find(V11 > EPSILON & V10 < -EPSILON);

theta2(set1) = theta(set1);
theta2(set2) = theta(set2)+180;
theta2(set3) = theta(set3)+180;
theta2(set4) = theta(set4)+360;

Orientation = theta2;
% =======================================================================

%% ========================Histogram==========================

M = Ebin; % for excitation, 6 if omitted
T = Obin; % for orientation, 8 if omitted
S = Sbin; % number of bins in subhistogram
C = M*S;

Cval = -pi/2:pi/C:pi/2; % Excitation
Cvalcen = (Cval(1:end-1)+Cval(2:end))/2;

Tval = 0:360/T:360; % Orientation
% Tvalcen = (Tval(1:end-1)+Tval(2:end))/2;

h2d = zeros(C,T);

% for i = 1:C
%     if i>1
%         temp = Orientation(Excitation>Cval(i)&Excitation<=Cval(i+1));
%     else
%         temp = Orientation(Excitation>=Cval(i)&Excitation<=Cval(i+1));
%     end
%     h2d(i,:) = hist(temp,Tvalcen);
% end
for i = 1:T
    if i>1
        temp = Excitation(Orientation>Tval(i)&Orientation<=Tval(i+1));
    else
        temp = Excitation(Orientation>=Tval(i)&Orientation<=Tval(i+1));
    end
    h2d(:,i) = hist(temp,Cvalcen);
end

h = h2d';
h = reshape(h,[T,S,M]);
temph = []; % final 1d histogram
for j = 1:M;
    temp = h(:,:,j);
    temph = [temph; temp(:)];
end
h1d = temph;

%=========================================================================
