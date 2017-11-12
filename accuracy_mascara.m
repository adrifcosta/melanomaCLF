% Outubro de 2016
% Base de dados e Análise de Informação
% Mestrado Integrado em Engenharia Biomédica
%
% Adriana Costa
% João Sousa
%
% FUNÇÃO QUE CALCULA A ACCURACY DA MÁSCARA OBTIDA PELAS NOSSOS MÉTODOS
% COMPARANDO-A COM A MÁSCARA GOLD STANDARD
function accuracy = accuracy_mascara(masc1,masc_gold)
    n_pixels = size(masc1,1) * size(masc1,2);
    contador1 = 0;
    %Comparar mascaras
    for linha = 1:size(masc1,1)
        for coluna = 1:size(masc1,2)
           if (masc1(linha,coluna) == masc_gold(linha,coluna))
               contador1 = contador1 + 1;
           end
        end
    end
    accuracy = (contador1 / n_pixels) * 100;
end