function result = CNN_features
outputFolder = [pwd '/matconvnet-1.0-beta23/CNN_features']; % define output folder

% Load Images
rootFolder = [outputFolder '/images'];
categories = {'benign', 'malignant'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

tbl = countEachLabel(imds)
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
imds = splitEachLabel(imds, minSetCount, 'randomize');
countEachLabel(imds)

benign = find(imds.Labels == 'benign', 1);
malignant = find(imds.Labels == 'malignant', 1);

figure
subplot(1,2,1);
imshow(readimage(imds,benign))
subplot(1,2,2);
imshow(readimage(imds,malignant))

cnnMatFile = ([pwd '/matconvnet-1.0-beta23/Nets/imagenet-caffe-alex.mat']);

% Load Pre-trained CNN
convnet = helperImportMatConvNet(cnnMatFile)

convnet.Layers

convnet.Layers(1)

convnet.Layers(end)

% Pre-process Images For CNN
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

function Iout = readAndPreprocessImage(filename)

    I = imread(filename);
    
    if ismatrix(I)
        I = cat(3,I,I,I);
    end
 
    Iout = imresize(I, [227 227]);  
end
[testSet, trainingSet] = splitEachLabel(imds, 0.3, 'randomize');

featureLayer = 'fc7';

trainingFeatures = activations(convnet, trainingSet, featureLayer,'MiniBatchSize', 32, 'OutputAs', 'columns');
trainingLabels = trainingSet.Labels;

% Extract test features using the CNN
testFeatures = activations(convnet, testSet, featureLayer, 'MiniBatchSize',32);
% Get the known labels
testLabels = testSet.Labels;

% Train A Multiclass SVM Classifier Using CNN Features
classifier = fitcecoc(trainingFeatures, trainingLabels,'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
classifier1 = fitcsvm(trainingFeatures', trainingLabels);
classifier2 = fitcknn(trainingFeatures', trainingLabels);
classifier3 = fitcnb(trainingFeatures', trainingLabels);
classifier4 = fitctree(trainingFeatures', trainingLabels);
classifier5 = TreeBagger(50,trainingFeatures', trainingLabels);
classifier7 = fitcdiscr(trainingFeatures', trainingLabels);
classifier8 = fitcensemble(trainingFeatures', trainingLabels);

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);
predictedLabels1 = predict(classifier1, testFeatures);
predictedLabels2 = predict(classifier2, testFeatures);
predictedLabels3 = predict(classifier3, testFeatures);
predictedLabels4 = predict(classifier4, testFeatures);
predictedLabels5 = categorical(predict(classifier5, testFeatures));
predictedLabels7 = predict(classifier7, testFeatures);
predictedLabels8 = predict(classifier8, testFeatures);

% Tabulate the results using a confusion matrix.
accuracys = zeros(2,8);

confMat = confusionmat(testLabels, predictedLabels);
confMats{1,1} = confMat;
accuracys(1,1) = confMat(2,2) * 100 / (confMat(2,2) + confMat(1,2));
accuracys(2,1) = confMat(1,1) * 100 / (confMat(1,1) + confMat(2,1));

confMat = confusionmat(testLabels, predictedLabels1);
confMats{1,2} = confMat;
accuracys(1,2) = confMat(2,2) * 100 / (confMat(2,2) + confMat(1,2));
accuracys(2,2) = confMat(1,1) * 100 / (confMat(1,1) + confMat(2,1));

confMat = confusionmat(testLabels, predictedLabels2);
confMats{1,3} = confMat;
accuracys(1,3) = confMat(2,2) * 100 / (confMat(2,2) + confMat(1,2));
accuracys(2,3) = confMat(1,1) * 100 / (confMat(1,1) + confMat(2,1));

confMat = confusionmat(testLabels, predictedLabels3);
confMats{1,4} = confMat;
accuracys(1,4) = confMat(2,2) * 100 / (confMat(2,2) + confMat(1,2));
accuracys(2,4) = confMat(1,1) * 100 / (confMat(1,1) + confMat(2,1));

confMat = confusionmat(testLabels, predictedLabels4);
confMats{1,5} = confMat;
accuracys(1,5) = confMat(2,2) * 100 / (confMat(2,2) + confMat(1,2));
accuracys(2,5) = confMat(1,1) * 100 / (confMat(1,1) + confMat(2,1));

confMat = confusionmat(testLabels, predictedLabels5);
confMats{1,6} = confMat;
accuracys(1,6) = confMat(2,2) * 100 / (confMat(2,2) + confMat(1,2));
accuracys(2,6) = confMat(1,1) * 100 / (confMat(1,1) + confMat(2,1));

confMat = confusionmat(testLabels, predictedLabels7);
confMats{1,7} = confMat;
accuracys(1,7) = confMat(2,2) * 100 / (confMat(2,2) + confMat(1,2));
accuracys(2,7) = confMat(1,1) * 100 / (confMat(1,1) + confMat(2,1));

confMat = confusionmat(testLabels, predictedLabels8);
confMats{1,8} = confMat;
accuracys(1,8) = confMat(2,2) * 100 / (confMat(2,2) + confMat(1,2));
accuracys(2,8) = confMat(1,1) * 100 / (confMat(1,1) + confMat(2,1));

result = {confMats,accuracys};
% Convert confusion matrix into percentage form
%confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

end