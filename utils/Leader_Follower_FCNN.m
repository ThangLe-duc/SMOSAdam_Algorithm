% Create and determine leader and follower of SMO-SAdam algorithm
function [dlnetL, dlnetF] = Leader_Follower_FCNN(XTrain,YTrain,height,width,channels,numClasses,classes)

% Specify data.
dlX = dlarray(single(XTrain),'SSCB');
Y = zeros(numClasses, 3000, 'single');
for c = 1:numClasses
    Y(c,YTrain==classes(c)) = 1;
end

% Create the first network.
fc1_1 = fullyConnectedLayer(1000,'Name','FC1','WeightsInitializer','he');
fc1_2 = fullyConnectedLayer(1000,'Name','FC2','WeightsInitializer','he');
fc1_3 = fullyConnectedLayer(numClasses,'Name','FC3','WeightsInitializer','he');
layers1 = [ ...
    imageInputLayer([height width channels],'Name','input','Mean',mean(XTrain,4));
    fc1_1;
    batchNormalizationLayer
    reluLayer('Name','relu1');
%     tanhLayer('Name','tanh1');
%     sigmoidLayer('Name','sigmoid1');
    fc1_2;
    batchNormalizationLayer
    reluLayer('Name','relu2');
%     tanhLayer('Name','tanh2');
%     sigmoidLayer('Name','sigmoid2');
    fc1_3;
    softmaxLayer('Name','softmax')];
lgraph1 = layerGraph(layers1);
dlnet1 = dlnetwork(lgraph1);

% Create the second network.
fc2_1 = fullyConnectedLayer(1000,'Name','FC1','WeightsInitializer','he');
fc2_2 = fullyConnectedLayer(1000,'Name','FC2','WeightsInitializer','he');
fc2_3 = fullyConnectedLayer(numClasses,'Name','FC3','WeightsInitializer','he');
layers2 = [ ...
    imageInputLayer([height width channels],'Name','input','Mean',mean(XTrain,4));
    fc2_1;
    batchNormalizationLayer
    reluLayer('Name','relu1');
%     tanhLayer('Name','tanh1');
%     sigmoidLayer('Name','sigmoid1');
    fc2_2;
    batchNormalizationLayer
    reluLayer('Name','relu2');
%     tanhLayer('Name','tanh2');
%     sigmoidLayer('Name','sigmoid2');
    fc2_3;
    softmaxLayer('Name','softmax')];
lgraph2 = layerGraph(layers2);
dlnet2 = dlnetwork(lgraph2);

% Evaluate networks
[~,loss1] = dlfeval(@modelGradients,dlnet1,dlX,Y);
[~,loss2] = dlfeval(@modelGradients,dlnet2,dlX,Y);

% Rank networks to determine the leader and the follower
if loss1 < loss2
    dlnetL = dlnet1;
    dlnetF = dlnet2;
else
    dlnetL = dlnet2;
    dlnetF = dlnet1;
end