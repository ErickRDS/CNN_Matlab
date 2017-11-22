%% Code to Create a Convolutional Neural Network for Image Recognition
%% Coded by:
%Name: Erick Ramos dos Santos
clear 
close all
clc
%Receiving the database
Data = fullfile('data_matlab'); %Here, you put the name of the folder
Data = imageDatastore(Data,'IncludeSubfolders',true,'LabelSource','foldernames');
CountLabel = Data.countEachLabel;
CountLabel = table2array(CountLabel(1,2));
%Some images in the database
figure (1)
quant = size(Data.Files,1);
datas = randperm(quant,20);
for i = 1:20
subplot(4,5,i);
imshow(Data.Files{datas(i)});
end
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
set(gcf, 'Name', 'Imagens Presentes no Banco de Treinamento da Primeira CNN', 'NumberTitle', 'Off')
drawnow;
clear datas i quant
%Defining how many images wuill be tested and how many will be trained.
%A good value is 75%
percent = 0.75;
trainingNumFiles = int64(percent*CountLabel);
rng(1)
[trainData,testData] = splitEachLabel(Data,trainingNumFiles,'randomize');
%Getting the sizes of the dataset
%Remember: All the images in the database must have the same dimensions
img = readimage(Data,1);
[height, width, numChannels, ~] = size(img);
imageSize = [height width numChannels];
clear img percent trainingNumFiles
%% Creating the CNN
%Feel free to change the values of filter size and number of filters
filterSize = [5 5];
numFilters = 32;
%Input Layer
inputLayer = imageInputLayer(imageSize);
%MIddle Layer
middleLayers = [
    %Feel free to put more convolution layers if you want
    convolution2dLayer(filterSize, numFilters, 'Padding', 2)
    reluLayer()
    %Test with meanPooling2dLayer too, just to compare and define what is
    %best to your database
    maxPooling2dLayer(3, 'Stride', 2)
    convolution2dLayer(filterSize, numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(3, 'Stride',2)
    convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(3, 'Stride',2)
    ];
%Output Layer
finalLayers = [
    %Feel free to put more fully connected layers if you want
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(2) %Here you put how many classes exist in the database
    softmaxLayer
    classificationLayer
    ];
%Joining the layers
layers = [
    inputLayer
    middleLayers
    finalLayers
    ];
%Options of training
options = trainingOptions('sgdm','MaxEpochs',50, ...
	'InitialLearnRate',0.0001);
convnet = trainNetwork(trainData,layers,options);
%Testing
YTest = classify(convnet,testData);
TTest = testData.Labels;
confMat = confusionmat(TTest, YTest);
helperDisplayConfusionMatrix(confMat)
accuracy = sum(YTest == TTest)/numel(TTest)
%Saving the network
save('CNN','convnet');
