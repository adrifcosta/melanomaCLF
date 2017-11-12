% Outubro de 2016
% Base de dados e Análise de Informação
% Mestrado Integrado em Engenharia Biomédica
%
% Adriana Costa
% João Sousa
%
% FUNÇÃO QUE APLICA O ACTIVE CONTOURS
function mascara_nossa = snakes(mascara_binaria,mascara_gold)
    mascara_nossa = zeros(size(mascara_binaria,1),size(mascara_binaria,2));
    maior_accuracy = 0;
    
    %tentar com diferentes tamanhos iniciais de mascara
    for i=4:3:7
        mask = zeros(size(mascara_binaria));
        linha = round(size(mascara_binaria,1)/i);
        coluna = round(size(mascara_binaria,2)/i);
        mask(linha:end-linha,coluna:end-coluna) = 1;
        bw = activecontour(mascara_binaria,mask,100);
        
        %analisar a qualidade da mascara
        accuracy = accuracy_mascara(bw,mascara_gold);
        if (maior_accuracy < accuracy)
           maior_accuracy = accuracy;
           mascara_nossa = bw;
        end
    end


end