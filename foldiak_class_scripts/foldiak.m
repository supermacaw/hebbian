% foldiak.m - simulates Foldiak's sparse coding circuit

batch_size =100;
num_trials = 1000;

% number of inputs
N=64;

% number of outputs
M=16;

% target output firing rate
p=0.2;

% Initialize network parameters (comment these lines out if you want 
% to restart the script from where you left off in a previous run)
%
% feedfoward weights
Q=rand(M,N);
Q=diag(1./sqrt(sum(Q.*Q,2)))*Q;
% horizontal connections
W=zeros(M);
% thresholds
theta=ones(M,1);


% learning rates
alpha=.1;
beta=.01;
gamma=.1;

% rate parameter for computing moving averages
eta_ave=0.1;

Y_ave=p;
Cyy_ave=p^2;

for t=1:num_trials
    
    % generate data for this batch
    X=gen_lines(N,batch_size,p);
    
    % compute outputs
    Y=sparsify_f(X,Q,W,theta);
    
    % compute statistics for this batch
    muy=mean(Y,2);
    Cyy=Y*Y'/batch_size;
    Cyx=Y*X'/batch_size;
    
    % update lateral weights
    dW=... % dW rule here
    W=W+dW;
    W=W-diag(diag(W)); % zero out diagonal
    W(find(W>0))=0;    % rectify weights

    % update feedforward weights
    dQ=... % dQ rule here
    Q=Q+dQ;

    % update thresholds
    dtheta=... % dtheta rule here (theta = t in Foldiak's paper)
    theta=theta+dtheta;
    
    % compute moving averages of muy and Cyy
    Y_ave=(1-eta_ave)*Y_ave + eta_ave*muy;
    Cyy_ave=(1-eta_ave)*Cyy_ave + eta_ave*Cyy;
    
    % display network state and activity statistics
    show_network(theta,Y_ave,W,Cyy_ave,Q);
    
end
