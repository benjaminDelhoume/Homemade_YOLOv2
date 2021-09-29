%Adapté et amélioré à partir du code de démonstration :
%openExample('deeplearning_shared/ObjectDetectionUsingYOLOV2DeepLearningExample')
%Benjamin Delhoume 4TCA

%Variable globale qui nous permet de définir si l'on utilise un réseau pré
%entrainé ou si on entraîne notre propre réseau.
doTraining = false;

%Télechargement du réseau pré-entraîné et du dataset de véhicule
if ~doTraining && ~exist('yolov2ResNet50VehicleExample_19b.mat','file')    
    disp('Downloading pretrained detector (98 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/yolov2ResNet50VehicleExample_19b.mat';
    websave('yolov2ResNet50VehicleExample_19b.mat',pretrainedURL);
end

%Après avoir téléchargé le dataset de test on le dézipe
unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;

% On affiche les premières lignes de notre dataset (à des fins de
% compréhension)
vehicleDataset(1:4,:)

% On ajoute le path de notre vehicleFolder
vehicleDataset.imageFilename = fullfile(pwd,vehicleDataset.imageFilename);

%Découpage de notre set de donnée en 60% d'entraînement, 10% validation et
%le reste en set de test.
%Si on veut le même tirage plusieurs fois mettre rng(0);
rng(randi(1000));
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * length(shuffledIndices) );

trainingIdx = 1:idx;
trainingDataTbl = vehicleDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = vehicleDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = vehicleDataset(shuffledIndices(testIdx),:);

% On utilise imageDatastore et boxLaberlDatastore pour créer des datastores
% ou l'on mets les images et labels pendant l'entraînement et la validation
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'vehicle'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));

%On combine les images et box labels datastores
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

%On affiche la première image de notre set d'entraînement et les boxs associé pour voir à quoi
%cela ressemble
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%Mise en place des données d'entrée, le nombre de classes et les anchorboxes
inputSize = [224 224 3];

numClasses = width(vehicleDataset)-1;

trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 7;


%On utilise la méthode matlab estimateAnchorBoxes pour estimer nos
%anchorboxes en fonction de la taille des objets du set de training
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

%On définit notre architecture comme l'architecture de resnet50
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';

lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
%Utilisation de la méthode Augmented Data pour rendre notre dataset
%d'entrainement plus complet
augmentedTrainingData = transform(trainingData,@augmentData);

% Boucle d'affichage d'une image du set d'entrainement puis de ses versions
% augmentées
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)

%Process de nos sets d'entraînement augmenté et de validation pour
%l'entraînement

preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
preprocessedValidationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(preprocessedTrainingData);

%Affichage d'une image utilisée pour l'entraînement
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)


%Simple affichage des couches de notre réseau pour analyse (commentable)
analyzeNetwork(lgraph);


%Les options modifiables, si l'on entraîne notre propre réseau avec
%l'architecture de resnet50
options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',16,...
        'Shuffle','every-epoch', ...
        'CheckpointPath',tempdir, ...
        'ValidationData',preprocessedValidationData,...
        'ValidationFrequency',40,...
        'Verbose',false,...
        'Plots','training-progress');
    
% Vérification de notre variable globale du début
if doTraining       
    % Entrainement de notre propre YOLO v2 détecteur
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % Utilisation du réseau pré entrainé
    pretrained = load('yolov2ResNet50VehicleExample_19b.mat');
    detector = pretrained.detector;
end


%Boucle pour l'affichage d'image aléatoire dans notre set de test la boucle
%peut être augmenté ou réduite (peut provoquer des erreurs non-bloquantes je
%travaille dessus)
for j = 1:3
    I = imread(testDataTbl.imageFilename{j});
    I = imresize(I,inputSize(1:2));
    [bboxes,scores] = detect(detector,I);
    I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
    figure
    imshow(I)
end
%On peut remplacer la boucle ci-dessus par une image choisis par nos soins
%mais il faut faire attention de la resize de la bonne taille


%Calcul et affichage du graph Précision/Recall pour l'efficacité de notre
%réseau
preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));

detectionResults = detect(detector, preprocessedTestData);

[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);


figure('Name','Precision')
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))

% A partir d'ici le code est récupérer des sources matlab pour la
% génération de code utilisable sur un GPU avec CUDA (non testé)


function B = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.
B = cell(size(A));

I = A{1};
sz = size(I);
if numel(sz)==3 && sz(3) == 3
    I = jitterColorHSV(I,...
        'Contrast',0.2,...
        'Hue',0,...
        'Saturation',0.1,...
        'Brightness',0.2);
end

% Randomly flip and scale image.
tform = randomAffine2d('XReflection',true,'Scale',[1 1.1]);
rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');
B{1} = imwarp(I,tform,'OutputView',rout);

% Apply same transform to boxes.
[B{2},indices] = bboxwarp(A{2},tform,rout,'OverlapThreshold',0.25);
B{3} = A{3}(indices);

% Return original data only when all boxes are removed by warping.
if isempty(indices)
    B = A;
end
end

function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to the targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
data{2} = bboxresize(data{2},scale);
end
