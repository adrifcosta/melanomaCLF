function metrics = classification_functions(features_manual_todas,labels_number,n)
if (n==1)
    treino_index = round(0.7 * size(features_manual_todas,1));
    %SVM
    SVModel = fitcsvm(features_manual_todas(1:treino_index,:),labels_number(1:treino_index,:),'Standardize',true,'KernelFunction','polynomial','Solver','ISDA','Cost',[0,1;6,0],'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('Optimizer','randomsearch','AcquisitionFunctionName','lower-confidence-bound'));
    classified_SVM = predict(SVModel,features_manual_todas(treino_index + 1: end,:));
%     save('SVModel','SVModel');

    %KNN
    KNNModel = fitcknn(features_manual_todas(1:treino_index,:),labels_number(1:treino_index,:),'NumNeighbors',5,'Distance','cosine','DistanceWeight','squaredinverse','Standardize',true,'Cost',[0,1;5,0],'NSMethod','exhaustive','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('Optimizer','bayesopt','AcquisitionFunctionName','expected-improvement'));
    classified_KNN = predict(KNNModel,features_manual_todas(treino_index+1:end,:));
%     save('KNNModel','KNNModel');


    %Naive-Bayes
    NBModel = fitcnb(features_manual_todas(1:treino_index,:),labels_number(1:treino_index,:),'DistributionNames','kernel','Cost',[0,1;6,0]);
    classified_NB = predict(NBModel,features_manual_todas(treino_index+1:end,:));
%     save('NBModel','NBModel');


    %Decision Tree
    TreeModel = fitctree(features_manual_todas(1:treino_index,:),labels_number(1:treino_index,:),'PredictorSelection','interaction-curvature','SplitCriterion','twoing','Cost',[0,1;5,0],'MinLeafSize',60);
    classified_Tree = predict(TreeModel,features_manual_todas(treino_index+1:end,:));
%     save('TreeModel','TreeModel');


    %Tree Bagger
    TBModel = TreeBagger(50,features_manual_todas(1:treino_index,:),labels_number(1:treino_index,:),'Cost',[0,1;9,0],'OOBPrediction','on');
    classified_TB1 = predict(TBModel,features_manual_todas(treino_index+1:end,:));
    classified_TB = zeros(length(classified_TB1),1);
    for i = 1 : length(classified_TB1)
       classified_TB(i,1) = str2double(classified_TB1{i}); 
    end
%     save('TBModel','TBModel');


    %LDA
    LDAModel = fitcdiscr(features_manual_todas(1:treino_index,:),labels_number(1:treino_index,:),'DiscrimType','diagquadratic');
    classified_LDA = predict(LDAModel,features_manual_todas(treino_index+1:end,:));
%     save('LDAModel','LDAModel');


    %Ensemble learner
    E1Model = fitcensemble(features_manual_todas(1:treino_index,:),labels_number(1:treino_index,:),'Method','RobustBoost','Cost',[0,1;3,0],'RobustMarginSigma',0.005);
    classified_E1 = predict(E1Model,features_manual_todas(treino_index+1:end,:));
%     save('E1Model','E1Model');
    
    results = [classified_SVM,classified_KNN,classified_NB,classified_Tree,classified_TB,classified_LDA,classified_E1];

    metrics = zeros(2,size(results,2));
    for i = 1:size(results,2)
        CM = confusionmat(labels_number(treino_index + 1:end),results(:,i));
        metrics(1,i) = CM(1,1) * 100 / (CM(1,1) + CM(2,1));
        metrics(2,i) = CM(2,2) * 100 / (CM(2,2) + CM(1,2));
    end


elseif (n==2) 
    benignos = randsample(find(labels_number == 1),length(find(labels_number == 1)));
    diff = length(find(labels_number == 1)) - length(find(labels_number == 2));

    labels_number2 = labels_number; 
    features_manual_todas2 = features_manual_todas;

    labels_number2(benignos(1:diff)) = [];
    features_manual_todas2(benignos(1:diff),:) = [];

    treino_index = round(0.7 * size(features_manual_todas2,1));
    %SVM
    SVModel = fitcsvm(features_manual_todas2(1:treino_index,:),labels_number2(1:treino_index,:),'Standardize',true,'KernelFunction','polynomial','Solver','L1QP','Cost',[0,1;1.5,0],'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('Optimizer','randomsearch','AcquisitionFunctionName','lower-confidence-bound'));
    classified_SVM = predict(SVModel,features_manual_todas2(treino_index + 1: end,:));

    %KNN
    KNNModel = fitcknn(features_manual_todas2(1:treino_index,:),labels_number2(1:treino_index,:),'NumNeighbors',5,'Distance','cosine','DistanceWeight','squaredinverse','Standardize',true,'Cost',[0,1;1.5,0],'NSMethod','exhaustive','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('Optimizer','bayesopt','AcquisitionFunctionName','expected-improvement'));
    classified_KNN = predict(KNNModel,features_manual_todas2(treino_index+1:end,:));

    %Naive-Bayes
    NBModel = fitcnb(features_manual_todas2(1:treino_index,:),labels_number2(1:treino_index,:),'DistributionNames','kernel','Kernel','box','Cost',[0,1;1.5,0]);
    classified_NB = predict(NBModel,features_manual_todas2(treino_index+1:end,:));

    %Decision Tree
    TreeModel = fitctree(features_manual_todas2(1:treino_index,:),labels_number2(1:treino_index,:),'MinLeafSize',30,'Cost',[0,1;1.5,0]);
    classified_Tree = predict(TreeModel,features_manual_todas2(treino_index+1:end,:));

    %Tree Bagger
    TBModel = TreeBagger(50,features_manual_todas2(1:treino_index,:),labels_number2(1:treino_index,:),'Cost',[0,1;1.5,0],'OOBPrediction','on');
    classified_TB1 = predict(TBModel,features_manual_todas2(treino_index+1:end,:));
    classified_TB = zeros(length(classified_TB1),1);
    for i = 1 : length(classified_TB1)
       classified_TB(i,1) = str2double(classified_TB1{i}); 
    end

    %LDA
    LDAModel = fitcdiscr(features_manual_todas2(1:treino_index,:),labels_number2(1:treino_index,:),'DiscrimType','diagquadratic');
    classified_LDA = predict(LDAModel,features_manual_todas2(treino_index+1:end,:));


    %Ensemble learner
    E1Model = fitcensemble(features_manual_todas2(1:treino_index,:),labels_number2(1:treino_index,:),'Method','RobustBoost','Cost',[0,1;1.5,0]);
    classified_E1 = predict(E1Model,features_manual_todas2(treino_index+1:end,:));
    
    results = [classified_SVM,classified_KNN,classified_NB,classified_Tree,classified_TB,classified_LDA,classified_E1];

    metrics = zeros(2,size(results,2));
    for i = 1:size(results,2)
        CM = confusionmat(labels_number2(treino_index + 1:end),results(:,i));
        metrics(1,i) = CM(1,1) * 100 / (CM(1,1) + CM(2,1));
        metrics(2,i) = CM(2,2) * 100 / (CM(2,2) + CM(1,2));
    end
    
    
end


end