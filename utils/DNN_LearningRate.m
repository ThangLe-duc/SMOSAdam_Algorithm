% Determine learning rate schedule
function [lrValue, lrInit, lrDropFrac, lrTepoch] = DNN_LearningRate(MaxEpoch, lrSchedule)
switch lrSchedule
    case 'none'
        lrValue = 1e-3;
        lrInit = lrValue;
        lrDropFrac = 0;
        lrTepoch = 200;
    case 'step'
        lrValue = 1e-3;
        lrInit = lrValue;
        lrDropFrac = 0.5;
        lrTepoch = 1000;
    case 'piecewise'
        lrValue = 1e-2;
        lrInit = lrValue;
        lrDropFrac = 0.5;
        lrTepoch = 2000;
end