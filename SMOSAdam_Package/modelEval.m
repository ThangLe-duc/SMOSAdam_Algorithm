% DNN forward mode
function loss = modelEval(dlnet,dlX,Y)
    dlYPred = forward(dlnet,dlX);
    loss = crossentropy(dlYPred,Y);
end