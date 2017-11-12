%% SEGMENTATION FINAL
% Outubro de 2016
% Base de dados e Análise de Informação
% Mestrado Integrado em Engenharia Biomédica
%
% Adriana Costa
% João Sousa
%
%
% O SCRIPT DEVE ESTAR NA MESMA DIRETORIA DA PASTA QUE CONTEM AS IMAGENS
% ABAIXO ESTÃO AS DIRETORIAS DAS IMAGENS DOS MELANOMA E RESPETIVAS MASCARAS
diretory1 = strcat(pwd,'/imagens_treino');
diretory2 = strcat(pwd,'/mascaras');

files1 = dir(diretory1);
files2 = dir(diretory2);
files1(1:3) = [];
files2(1:3) = [];
contador1 = 0;

% accuracys = zeros(900,14);
%features_imagens_manual = zeros(900,92);

%all_names_accuracys=[];
names_accuracys=[];
features_imagens_manual = zeros(900,92);

matriz_labels = load('labels.mat');
labels = matriz_labels.labels;
labels_values = zeros(900,1);

%Por os labels com 1 -> benign 2 -> malignant
for i = 1:900
   if isequal(labels{i,2},'malignant')
       labels{i,2} = 2;
       labels_values(i,1) = 2;
   end
end
for i = 1:900
    if isequal(labels{i,2},'benign')
        labels{i,2} = 1;
        labels_values(i,1) = 1;
    end
end

%Ciclo principal
for i = 1:size(files1,1) %size(files1,1)
    contador1=contador1+1;
    string1 = files1(i).name;
    string2 = files2(i).name;
    
    imagem = imread(strcat(diretory1,'/',string1));
    mascara = imbinarize(imread(strcat(diretory2,'/',string2)));
    
    % CASO A IMAGEM CONTENHA UMA MOLDURA CIRCULAR PRETA, PASSAMOS ESTA
    % MOLDURA PARA BRANCO DE MODO A QUE, AQUANDO DA SEGMENTAÇÃO, A MOLDURA
    % NÃO SEJA VISTA COMO UM SINAL
    levelr=0.09;
    levelg=0.09;
    levelb=0.09;
    
    i1=imbinarize(imagem(:,:,1), levelr);
    i2=imbinarize(imagem(:,:,2), levelg);
    i3=imbinarize(imagem(:,:,3), levelb);
    
    bw=(i1&i2&i3);
    bw=bwfill(bw,'holes');
    imagem1=rgb2gray(imagem);
    imagem1=imadjust(imagem1);
    for j=1:size(imagem1,1)
        for k=1:size(imagem1,2)
            if bw(j,k)==0
                imagem(j,k,:)=255;
            end
        end
    end
    
    
    % PREPROCESSAMENTO E SEGMENTAÇAO DA IMAGEM
    pre_processed = pre_processing_functions(imagem,mascara);
    mascara_final = pre_processed{1};
    %accuracys(contador1,1:14) = pre_processed{2};
    names_accuracys=[string(string1) pre_processed{2}];
    %all_names_accuracys=[all_names_accuracys ; names_accuracys];
    mascara_final = imfill(mascara_final,'holes');
    %imwrite(mascara_final,char(strcat('mascara_final_',string(string1))))
    
%     figure(1)
%     subplot(2,2,1), imshow(mascara_final), title('Máscara Final')
%     subplot(2,2,2), imshow(mascara), title('Máscara Gold Standard')
%     subplot(2,2,3), imshow(imagem), title('Imagem Pré-Processada')
    
    lesao_segmentada = bsxfun(@times, imagem, cast(mascara_final, 'like', imagem));
    %imwrite(lesao_segmentada,char(strcat('lesao_segmentada_',string(string1))))
    %Benign
%     if labels{i,2} == 1
%         imwrite(lesao_segmentada,char(strcat('lesao_segmentada_',string(string1))))
%     end
    %Malignant
%     if labels{i,2} == 2
%         imwrite(lesao_segmentada,char(strcat('lesao_segmentada_',string(string1))))
%     end
    %exterior_segmentado = bsxfun(@times, imagem, cast(~mascara_final, 'like', imagem));
    %imwrite(exterior_segmentado,char(strcat('exterior_segmentado_',string(string1))))
    %     subplot(2,2,4), imshow(lesao_segmentada), title('Imagem Segmentada')
    
    % FEATURE EXTRACTION

    manual_features = feature_extraction_functions(mascara_final,lesao_segmentada,exterior_segmentado);
    features_imagens_manual(i,1:92) = manual_features; 
    
end

%% FEATURE SELECTION
features_imagens_manual2 = feature_selection_functions(3,20,features_imagens_manual,labels);


%% CLASSIFICATION
metrics1 = classification_functions(features_imagens_manual,labels,1);


metrics2 = classification_functions(features_imagens_manual,labels,2);


