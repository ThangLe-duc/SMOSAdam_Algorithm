%% Main function for implementing SMO-SAdam optimizer to solve MNIST classification problem
%% Programmer: Thang Le-Duc
%  Emails: le.duc.thang0312@gmail.com

%% Begin main function
clear all, close all, clc
seed = 10;       
rand('twister',seed);
randn('state',seed);
addpath('./SMOSAdam_Package')

%% Preprocess training data
TrainImgPath = '.\Datasets\MNIST\train-images.idx3-ubyte';
TrainLabPath = '.\Datasets\MNIST\train-labels.idx1-ubyte';
TestImgPath = '.\Datasets\MNIST\t10k-images.idx3-ubyte';
TestLabPath = '.\Datasets\MNIST\t10k-labels.idx1-ubyte';
[XTrain, YTrain] = readMNIST(TrainImgPath, TrainLabPath, 60000, 0);
[XTest, YTest] = readMNIST(TestImgPath, TestLabPath, 10000, 0);
XTrain = reshape(XTrain, size(XTrain, 1), size(XTrain, 2), 1, size(XTrain, 3));
XTest = reshape(XTest, size(XTest, 1), size(XTest, 2), 1, size(XTest, 3));

YTrain = categorical(YTrain);
YTest = categorical(YTest);
[height,width,channels,num_samples] = size(XTrain);
classes = categories(YTrain);
numClasses = numel(classes);

XTrain = XTrain./255;
XTest = XTest./255;

%% Specify training options
miniBatchSize = 120;
numEpochs = 20;
numObservations = numel(YTrain);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
lrSchedule = 'none';    % 'none' 'step' 'piecewise'
[lr, lrInit, lrDropFrac, Tepoch] = DNN_LearningRate(numEpochs, lrSchedule);
% Specify training on a GPU or not
executionEnvironment = "cpu"; % "auto" "gpu" "cpu"
XTest = double(XTest);
dlXTest = dlarray(XTest,'SSCB');
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlXTest = gpuArray(dlXTest);
end
Yall = zeros(numClasses, size(YTest,1), 'single');
for c = 1:numClasses
    Yall(c,YTest==classes(c)) = 1;
end
% Initialize the parameters for the SMO-SAdam solver
averageGrad = []; averageSqGrad = []; averageSqGradPre = [];
avg_gsqPreL = []; vtplus_pLPre = []; vt_curPre = [];
% Define gradients function of DNN model
accfun = dlaccelerate(@modelGradients);
% Visualize the training progress in a plot
plots = "training-progress";
if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end

%% Train network.
Ttime = 1;
iteration = 0;
lossTrain = zeros(numEpochs*numIterationsPerEpoch,Ttime);
accuracyTrain = zeros(numEpochs*numIterationsPerEpoch,Ttime);
lossTest = zeros(numEpochs,Ttime);
accuracyTest = zeros(numEpochs,Ttime);
for time = 1:Ttime
% Define the network architecture and specify the leader and the follower
[dlnetL, dlnetF] = Leader_Follower_Nets(XTrain(:,:,:,1:3000),YTrain(1:3000),height,width,channels,numClasses,classes);
% Start training process
start = tic;
for epoch = 1:numEpochs
    % Shuffle data.
    idx = randperm(numel(YTrain));
    XTrain = XTrain(:,:,:,idx);
    YTrain = YTrain(idx);
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        % Read mini-batch of data and convert the labels to dummy variables
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        X = XTrain(:,:,:,idx);
        Y = zeros(numClasses, miniBatchSize, 'single');
        for c = 1:numClasses
            Y(c,YTrain(idx)==classes(c)) = 1;
        end
        % Convert mini-batch of data to a dlarray
        dlX = dlarray(single(X),'SSCB');
        % Convert data to a gpuArray if needed
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        % Update learning rate
        lr = LRSchedule(lrInit, lrDropFrac, Tepoch, epoch, lrSchedule);
        % Update the network parameters using the SMO-SAdam optimizer
        [gradientsL,lossTrain(iteration,time)] = dlfeval(accfun,dlnetL,dlX,Y);
        [dlnetL.Learnables,dlnetF.Learnables,averageGrad,averageSqGrad,averageSqGradPre,avg_gsqPreL,vtplus_pLPre,vt_curPre] = ...
            smosadamupdate(dlnetL.Learnables,dlnetF.Learnables,gradientsL,averageGrad,averageSqGrad,averageSqGradPre,...
            avg_gsqPreL,vtplus_pLPre,vt_curPre,iteration,epoch,lr);
        % Evaluate the training classification accuracy
        dlY = forward(dlnetL,dlX);
        [~,idxIter] = max(extractdata(dlY),[],1);
        YPredTrain = classes(idxIter);
        accuracyTrain(iteration,time) = mean(YPredTrain==YTrain(idx));
        % Display the training progress
        if plots == "training-progress"
            D(time) = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(lossTrain(iteration,time))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
    accTrain = mean(accuracyTrain((epoch-1)*numIterationsPerEpoch+1 : epoch*numIterationsPerEpoch,time));
    % Calculate test loss and classification accuracy
    lossTest(epoch,time) = modelEval(dlnetL,dlXTest,Yall);
    dlYPred = forward(dlnetL,dlXTest);
    [~,idx] = max(extractdata(dlYPred),[],1);
    YPred = classes(idx);
    accuracyTest(epoch,time) = mean(YPred==YTest);
    accTest = accuracyTest(epoch,time);
    fprintf('Mean training and test errors are %2.4f and %2.4f at epoch %i \n',100*(1-accTrain),100*(1-accTest),epoch);
end
iteration = 0;
fprintf('--------------------------------End of training process---------------------------------- \n');
end

save DNN.mat dlnetL lr miniBatchSize num_samples numEpochs 
save results.mat lossTrain lossTest accuracyTrain accuracyTest YPred YTest D