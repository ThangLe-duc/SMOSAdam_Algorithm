function [p, f, avg_g, avg_gsq, gCorPre, avg_gsqPreL, vtplus_pLPre, vt_curPre] = smosadamupdate(p, f, g, avg_g, avg_gsq,...
    gCorPre, avg_gsqPreL, vtplus_pLPre, vt_curPre, t, epoch, lr, beta1, beta2, epsilon)
%   SMOSADAMUPDATE Update parameters via sequential motion with adaptive moment estimation
%
%   [L_NET,F_NET,AVG_G,AVG_SQG,GRADCOR_PRE,AVG_SQG_PRE,VEL_2ND_PRE,VEL_3RD_PRE] = 
%   SMOSADAMUPDATE(L_NET,F_NET,GRAD,AVG_G,AVG_SQG,GRADCOR_PRE,AVG_SQG_PRE,
%   VEL_2ND_PRE,VEL_3RD_PRE,ITER,EPOCH,LR,BETA1,BETA2,EPS) updates
%   the learnable parameters of the dlnetwork of the leader and the follower 
%   using the SMO-SAdam gradient descent algorithm, with a default global 
%   learning rate of 0.001, a gradient decay factor of 0.9 and a squared 
%   gradient decay factor of 0.999. The default regularization coefficient is 
%   set at 0.0001. Use SMOSADAMUPDATE to iteratively update learnable parameters
%   during network training.
%
%   Input GRAD contains the gradients of the loss with respect to each of
%   the network parameters. Inputs AVG_G and AVG_SQG contain, the moving
%   average of the parameter gradients and the moving average of the
%   element-wise squares of the parameter gradients, respectively. AVG_SQG_PRE
%   contains the moving average of the element-wise squares of the previous 
%   leader movement. GRADCOR_PRE, VEL_2ND_PRE and VEL_3RD_PRE contain the 
%   gradient correction, the velocity update at 2nd and 3rd stages of the
%   follower movement.
%   AVG_G, AVG_SQG,GRADCOR_PRE,AVG_SQG_PRE,VEL_2ND_PRE,VEL_3RD_PRE are obtained 
%   from the previous call to SMO-SAdam and they must be tables with the same structure as
%   NET.Learnables, with a Value variable containing a cell array of
%   parameter gradients, average gradients, or average squared gradients.
%   Input GRAD can be obtained using the dlgradient and dlfeval functions.
%   The global learning rate is multiplied by the corresponding learning
%   rate factor for each parameter in each layer of the dlnetwork. 
%
%   Input ITER contains the update iteration number. ITER must be a
%   positive integer. Use a value of 1 for the first call to SMOSADAMUPDATE
%   and increment by 1 for each successive call in a series of iterations. The
%   SMO-SAdam algorithm uses this value to correct for bias in the moving
%   averages at the beginning of a set of iterations.
%
%   Outputs L_NET, F_NET are the updated dlnetworks of the leader and the 
%   follower, respectively. L_NET, F_NET can be a dlarray, a numeric array,
%   a cell array, a structure, or a table with a Value variable containing 
%   the learnable parameters of the network. GRAD, AVG_G, AVG_SQG,GRADCOR_PRE,
%   AVG_SQG_PRE,VEL_2ND_PRE,VEL_3RD_PRE must have the same datatype and 
%   ordering as L_NET, F_NET.  All parameter values are updated
%   using the global learning rate.
%
%   EPSILON specifies a small constant used to prevent division by zero in 
%   the update equation. The default value of EPSILON is 1e-8.
%
%   Copyright 2023 Thang Le-Duc, email: le.duc.thang0312@gmail.com

arguments
    p
    f
    g
    avg_g
    avg_gsq
    gCorPre
    avg_gsqPreL
    vtplus_pLPre
    vt_curPre
    t(1,1) {mustBeNumeric, mustBePositive, mustBeInteger}
    epoch(1,1) {mustBeNumeric, mustBePositive, mustBeInteger}
    
    lr(1,1) {mustBeNumeric, mustBeFinite, mustBeNonnegative} = 0.001;
    beta1(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta1, 0), mustBeLessThan(beta1, 1)} = 0.9;
    beta2(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta2, 0), mustBeLessThan(beta2, 1)}= 0.999;
    epsilon(1,1) {mustBeNumeric, mustBeFinite, mustBePositive} = 1e-8;
end

persistent func
if isempty(func)
    func = deep.internal.LearnableUpdateFunction( ...
        @iSingleStepValue, ...
        @iSingleStepParameter );
end

if isempty(avg_g) && isempty(avg_gsq)
    % Execute a first-step update with g_av and g_sq_av set to 0.  The step
    % will create arrays for these that are the correct size
    paramArgs = {f, g};
    fixedArgs = {0, 0, 0, 0, 0, 0, t, epoch, lr, beta1, beta2, epsilon};
else
    % Execute the normal update
    paramArgs = {f, g, matlab.lang.internal.move(avg_g), matlab.lang.internal.move(avg_gsq), gCorPre, avg_gsqPreL, vtplus_pLPre, vt_curPre};
    fixedArgs = {t, epoch, lr, beta1, beta2, epsilon};
end

[p, f, avg_g, avg_gsq, gCorPre, avg_gsqPreL, vtplus_pLPre, vt_curPre] = deep.internal.networkContainerFixedArgsFun(func, ...
    p, matlab.lang.internal.move(paramArgs), fixedArgs);

end

function [p, f, avg_gL, avg_gsqL, gCorPre, avg_gsqPreL, vtplus_pLPre, vt_curPre] = iSingleStepParameter(p, f, gL, avg_gL, avg_gsqL, gCorPre,...
    avg_gsqPreL, vtplus_pLPre, vt_curPre, t, epoch, lr, beta1, beta2, epsilon)
decayGrad = 0.1; RpLpF = 0.01; RpFLp = 0.01; RegCoef = 1e-4;
% Apply per-parameter learn-rate factor
lr = lr .* p.LearnRateFactor;

% Apply a correction factor due to the trailing averages being biased
% towards zero at the beginning.  This is fed into the learning rate
biasCorrection = sqrt(1-beta2.^t)./(1-beta1.^t);
effectiveLearnRate = biasCorrection.*lr;

%% Save leader and follower at previous iteration
pPre = p;
fPre = f;

%% Leader movement and follower update
[step, avg_gL, avg_gsqL] = nnet.internal.cnn.solver.adamstep(...
    gL, avg_gL, avg_gsqL, effectiveLearnRate, beta1, beta2, epsilon);
v = p.Value;
p.Value = [];
v = v + step;
% Determine IO and update follower for next iteration
f.Value = v;
pIO = v;

%% Follower movement and leader update
% Movement at 1st stage
k1 = max(2*rand-1-1e-8,-1+1e-8);
pL = pPre - RpLpF*k1*(pPre - fPre);
k2 = max(2*rand-1-1e-8,-1+1e-8);
pLIO = pIO - RpLpF*k2*(pIO - pL);

% Gradient correction
CorCoef = (1/(1+decayGrad*(epoch-1)))*tanh(0.0003*t).*(1 + RpLpF.*k2 - RpFLp.*k1.*RpLpF.*k2);
gCor = CorCoef*gL;

% Movement at 2nd stage
mtplus_pL = beta1*avg_gL + (1-beta1)*gL./(1-RpFLp.*k1);
vtplus_pL = beta2*avg_gsqL + (1-beta2)*(gL./(1-RpFLp.*k1)).^2;
vtplus_pL = max(vtplus_pLPre, vtplus_pL);
vtplus_pLPre = vtplus_pL;

% Movement at 3rd stage
mt_cur = beta1*mtplus_pL + (1-beta1)*gCor;
vt_cur = beta2*vtplus_pL + (1-beta2)*(gCor.^2);
vt_cur = max(vt_curPre, vt_cur);
vt_curPre = vt_cur;

% Calculate bias-corrected mt and vt
mt_cur_cor = mt_cur./(1-beta1^t);
vt_cur_cor = vt_cur./(1-beta2^t);

% Update leader for next iteration
p.Value = pLIO - lr*(mt_cur_cor)./(sqrt(vt_cur_cor)+1e-8) - lr*RegCoef*pLIO;

gCorPre = gCor;
end


function [p, f, avg_gL, avg_gsqL, gCorPre, avg_gsqPreL, vtplus_pLPre, vt_curPre] = iSingleStepValue(p, f, gL, avg_gL, avg_gsqL, gCorPre,...
    avg_gsqPreL, vtplus_pLPre, vt_curPre, t, epoch, lr, beta1, beta2, epsilon)
decayGrad = 0.1; RpLpF = 0.01; RpFLp = 0.01; RegCoef = 1e-4;
% Apply a correction factor due to the trailing averages being biased
% towards zero at the beginning.  This is fed into the learning rate
biasCorrection = sqrt(1-beta2.^t)./(1-beta1.^t);
effectiveLearnRate = biasCorrection.*lr;

%% Save leader and follower at previous iteration
pPre = p;
fPre = f;

%% Leader movement and follower update
[step, avg_gL, avg_gsqL] = nnet.internal.cnn.solver.adamstep(...
    gL, avg_gL, avg_gsqL, effectiveLearnRate, beta1, beta2, epsilon);
p = p + step;
% Determine IO and update follower for next iteration
pIO = p;
f = pIO;

%% Follower movement and leader update
% Movement at 1st stage
k1 = max(2*rand-1-1e-8,-1+1e-8);
pL = pPre - RpFLp*k1*(pPre - fPre);
k2 = max(2*rand-1-1e-8,-1+1e-8);
pLIO = pIO - RpLpF*k2*(pIO - pL);

% Gradient correction
CorCoef = (1/(1+decayGrad*(epoch-1)))*tanh(0.0003*t).*(1 + RpLpF.*k2 - RpFLp.*k1.*RpLpF.*k2);
gCor = CorCoef*gL;

% Movement at 2nd stage
mtplus_pL = beta1*avg_gL + (1-beta1)*gL./(1-RpFLp.*k1);
vtplus_pL = beta2*avg_gsqL + (1-beta2)*(gL./(1-RpFLp.*k1)).^2;
vtplus_pL = max(vtplus_pLPre, vtplus_pL);
vtplus_pLPre = vtplus_pL;

% Movement at 3rd stage
mt_cur = beta1*mtplus_pL + (1-beta1)*gCor;
vt_cur = beta2*vtplus_pL + (1-beta2)*(gCor.^2);
vt_cur = max(vt_curPre, vt_cur);
vt_curPre = vt_cur;

% Calculate bias-corrected mt_cur and vt_cur
mt_cur_cor = mt_cur./(1-beta1^t);
vt_cur_cor = vt_cur./(1-beta2^t);

% Update leader for next iteration
p = pLIO - lr*(mt_cur_cor)./(sqrt(vt_cur_cor)+1e-8) - lr*RegCoef*pLIO;

gCorPre = gCor;
end
