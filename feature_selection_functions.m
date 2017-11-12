% Outubro de 2016
% Base de dados e Análise de Informação
% Mestrado Integrado em Engenharia Biomédica
%
% Adriana Costa
% João Sousa
%
% FUNÇÃO QUE SELECIONA UM DETERMINADO NUMERO DE FEATURES
function features_novas = feature_selection_functions(tecnica,n_features,features,labels_number)
    %Feature selection
%PCA
%escolher quantas features queremos usar (neste caso 5)
    if (tecnica == 1)
        %PCA
        [coeff, dados_novos, latent] = pca(features);
        %escolher quantas features queremos usar (neste caso 5)
        dados_novos = dados_novos(:,1:n_features);
    elseif (tecnica == 2)
        %MDS
        dissimilaridade = squareform(pdist(features,'euclidean'));
        %preencher NaN por 0
        dissimilaridade(isnan(dissimilaridade)) = 0;
        [dados_novos,stress,disp] = mdscale(dissimilaridade,n_features,'criterion','sstress');
    elseif (tecnica == 3)
        %RankFeatures
        [IDX, Z] = rankfeatures(features', labels_number');
        dados_novos = zeros(size(features,1),n_features);
        for i = 1:n_features
            dados_novos(:,i) = features(:,IDX(i)) ;
        end
    end
    
    features_novas = dados_novos;
end