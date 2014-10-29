% sparsify_f.m - compute outputs of Foldiak network
%
% function Y=sparsify_f(X,Q,W,theta)
%
% X:        input array
% Q:        feedforward weights
% W:        horizontal weights
% theta:    thresholds
%
% Y:        outputs

function Y=sparsify_f(X,Q,W,theta,display_p)

if ~exist('display_p','var')
    display_p=0;
end

% size of data array
[N batch_size]=size(X);
sz=sqrt(N);

% number of outputs
M=size(Q,1);

% number iterations
num_iterations=50;
% rate parameter
eta=0.1;

% feedforward input
B=Q*X;
% threshold
T=repmat(theta,1,batch_size);

% initialize y*
Ys=zeros(M,batch_size);

if display_p
    figure(2)
    subplot(211)
    hy=bar(Ys(:,1)); axis([0 M+1 0 1])
    subplot(212)
    imagesc(reshape(X(:,1),sz,sz),[0 1]); axis image
end

for t=1:num_iterations
    
    % diffeq for y*
    Ys=(1-eta)*Ys+eta*sigmoid(B+W*Ys-T);
    
    if display_p
        set(hy,'YData',Ys(:,1))
        drawnow
    end
    
end

% threshold outputs to 0 or 1
Y=zeros(M,batch_size);
Y(find(Ys>0.5))=1;

if display_p
    set(hy,'YData',Y(:,1))
    drawnow
end
