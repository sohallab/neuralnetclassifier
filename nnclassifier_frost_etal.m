function [error, perf, fullclass, Win, Wout] = nnclassifier3(raster, labels, trainframes, testframes, Nruns, Nhidden, connprob, Npasses, learnrate, minact)

% input arguments specify which frames will be used for training vs. testing

% raster is the input raster; the number of rows = number of neurons and
% the number of columns = the number of timepoints

% labels = vector of labels (0s and 1s) that represent the target outputs
% for the classifier

% trainframes = indices of the frames to use for training

% testframes = indices of the frames to use for testing

% Nruns is the number of times to run this; each run will randomly hold
% out a different subset of data, and use different random weights from the
% input layer to the hidden layer

% Nhidden is the number of units in the hidden layer

% connprob is the probability of a connection from each inputs neuron to each hidden layer neuron

% Npasses is the number of passes through the data to do for training

% learnrate is the learning rate parameter which specifies how quickly the
% weights are updated

% minact are the minimum number of active neurons in each frame

% error is the average squared error for the training set

% perf is the performance, i.e., the fraction of the training set that is
% correctly classified as 0 or 1 (using 0.5 as a threshold)

% fullclass is a vector specifying the output (between 0 and 1) for each of
% the testing frames

% first identify frames that meet the threshold levels of activity

% also outputs weights and connections

act = sum(raster);
in = find(act>=minact);

%for raster
raster2 = raster(:,in);
%for labels
labels2 = labels(in);
%for train vs test
tmp = zeros(size(raster,2),1);
tmp(trainframes) = 1;
tmp(testframes) = 0;
tmp = tmp(in);
intrain = find(tmp == 1);
intest = find(tmp ==0);


N = size(raster2);
ncells = N(1);
ntimepoints = N(2);
ntrainpoints = numel(intrain);

for n=1:Nruns,
    
    % randomly set up weights from the input neurons to the hidden layer
    R = rand(ncells, Nhidden);
    Win = R < connprob;
    
    % randomize the order in which frames are visited
    Rs = rand(Npasses,ntrainpoints);
    
    Wout = zeros(Nhidden,1);

    % calculate the activity of each hidden unit for each frame
    for j=1:Nhidden,
        in2 = find(Win(:,j));
        x = sum(raster2(in2,:));
        hiddenact(:,j) = x';
    end
        
    for i=1:Npasses,
        [Y,order] = sort(Rs(i,:)); 
        order = intrain(order); 
        for j=1:ntrainpoints,
            % calculate the output
            x = sum(Wout' .* hiddenact(order(j),:));
            out = 1 / (1 + exp(-x));
            
            % calculate the error
            errortmp = out - labels2(order(j));
            
            % update the weights
            Wout = Wout - learnrate*out*(1-out)*errortmp*hiddenact(order(j),:)';
        end
    end
    
    % compute the performance on the testing data
    
    threshold = 0.5;
    for i=1:length(intest),
        outvect(i) = 1 / (1+exp(-sum(Wout' .* hiddenact(intest(i),:))));
    end

  
    fullclass = [outvect;labels2(intest)']; 
    error(n) = mean((outvect' - labels2(intest)) .* (outvect' - labels2(intest)));
    perf(n) = mean((outvect' > threshold) == labels2(intest));
   
end
