function value = test_cnn(net_finetuned,vetor_gt,dim)

%Use the images in the folder 'testar' 
files = dir([pwd '/imagens_treino/testar']);
files(1:3) = [];
scored_labels1 = zeros(1,length(files));
features_total = zeros(135,91);
for i=1:size(files,1)
   imagem = imread([pwd '/imagens_treino/testar/' files(i).name]); 
   value = inference_classification(imresize(imagem,[dim dim]),net_finetuned);
   scored_labels1(1,i) = value{1};
%    features_total(i,:) = value{2};
end

%Calculate metrics
if (size(scored_labels1,2) == size(vetor_gt,2))
    TP = 0;
    FP = 0;
    TN = 0;
    FN = 0;
    for i = 1:size(vetor_gt,2)
       if (scored_labels1(1,i) == 1 && vetor_gt(1,i) == 1)
           TN = TN + 1;
       elseif (scored_labels1(1,i) == 1 && vetor_gt(1,i) == 2)
           FN = FN + 1;
       elseif (scored_labels1(1,i) == 2 && vetor_gt(1,i) == 2)
           TP = TP + 1;
       elseif (scored_labels1(1,i) == 2 && vetor_gt(1,i) == 1)
           FP = FP + 1;
       end
    end

    accuracy_benigno = TN / (TN + FN);
    accuracy_maligno = TP / (TP + FP);
    jaccard_index = TP / (TN + FN + FP);
    
    value = {accuracy_benigno, accuracy_maligno, jaccard_index, features_total};
else
    error('Scored Labels has wrong size');
end