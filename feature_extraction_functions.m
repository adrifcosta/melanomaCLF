% Outubro de 2016
% Base de dados e Análise de Informação
% Mestrado Integrado em Engenharia Biomédica
%
% Adriana Costa
% João Sousa
%
%
function features = feature_extraction_functions(mascara_final,lesao_segmentada,exterior_segmentado)
    features = zeros(1,92);
%     auto_features=[];
    
    for i = 1:3
        if (i == 2)
           lesao_segmentada = rgb2hsv(lesao_segmentada);
        elseif (i == 3)
            lesao_segmentada = rgb2ntsc(lesao_segmentada);
        end
        
        temp = reshape(lesao_segmentada,[],3);
        temp1 = temp(:,1);
        temp2 = temp(:,2);
        temp3 = temp(:,3);
        condicao = (temp(:,1)~=0 & temp(:,2)~=0 & temp(:,3)~=0);
        lesao_segmentada_3_colunas = [temp1(condicao) temp2(condicao) temp3(condicao)];
        temp = reshape(exterior_segmentado,[],3);
        temp1 = temp(:,1);
        temp2 = temp(:,2);
        temp3 = temp(:,3);
        condicao = (temp(:,1)~=0 & temp(:,2)~=0 & temp(:,3)~=0);
        exterior_segmentado_3_colunas = [temp1(condicao) temp2(condicao) temp3(condicao)];

        pixel_lesao = size(lesao_segmentada_3_colunas,1);
        pixel_exterior = size(exterior_segmentado_3_colunas,1);
        pixel_imagem = pixel_lesao + pixel_exterior;
    
    
        %desvios padrao dos canais na lesao
        sd_lesao_red = 0;
        sd_lesao_green = 0;
        sd_lesao_blue = 0;
%         sd_ext_red = 0;
%         sd_ext_green = 0;
%         sd_ext_blue = 0;

        %medias dos canais na lesao/exterior
        red_mean_lesao = mean(lesao_segmentada_3_colunas(:,1));
        green_mean_lesao = mean(lesao_segmentada_3_colunas(:,2));
        blue_mean_lesao = mean(lesao_segmentada_3_colunas(:,3));
        RGB_mean_lesao = sqrt((red_mean_lesao)^2 + (green_mean_lesao)^2 + (blue_mean_lesao)^2); %somar os canais

%         red_mean_ext = mean(exterior_segmentado_3_colunas(:,1));
%         green_mean_ext = mean(exterior_segmentado_3_colunas(:,2));
%         blue_mean_ext = mean(exterior_segmentado_3_colunas(:,3));

        %sd lesao
        for j = 1:size(lesao_segmentada_3_colunas,1)
            sd_lesao_red = sd_lesao_red + (double(lesao_segmentada_3_colunas(j,1)) - red_mean_lesao)^2;
            sd_lesao_green = sd_lesao_green + (double(lesao_segmentada_3_colunas(j,2)) - green_mean_lesao)^2;
            sd_lesao_blue = sd_lesao_blue + (double(lesao_segmentada_3_colunas(j,3)) - blue_mean_lesao)^2;
        end

        %sd exterior
%         for i = 1:size(exterior_segmentado_3_colunas,2)
%             sd_ext_red = sd_ext_red + (double(exterior_segmentado_3_colunas(i,1)) - red_mean_ext)^2;
%             sd_ext_green = sd_ext_green + (double(exterior_segmentado_3_colunas(i,2)) - green_mean_ext)^2;
%             sd_ext_blue = sd_ext_blue + (double(exterior_segmentado_3_colunas(i,3)) - blue_mean_ext)^2;
%         end

        %desvio padrao dos canais da lesao/exterior
        sd_lesao_red = sqrt(sd_lesao_red / (pixel_lesao - 1));
        sd_lesao_green = sqrt(sd_lesao_green / (pixel_lesao - 1));
        sd_lesao_blue = sqrt(sd_lesao_blue / (pixel_lesao - 1));
        RGB_sd_lesao = sqrt((sd_lesao_red)^2 + (sd_lesao_green)^2 + (sd_lesao_blue)^2);

%         sd_ext_red = sqrt(sd_ext_red / (pixel_exterior - 1));
%         sd_ext_green = sqrt(sd_ext_green / (pixel_exterior - 1));
%         sd_ext_blue = sqrt(sd_ext_blue / (pixel_exterior - 1));

        %skewness dos canais da lesao/exterior
        skewnesses_lesao = skewness(double(lesao_segmentada_3_colunas));
%       skewnesses_exterior = skewness(double(exterior_segmentado_3_colunas));

        skew_lesao_red = skewnesses_lesao(1);
        skew_lesao_green = skewnesses_lesao(2);
        skew_lesao_blue = skewnesses_lesao(3);
        RGB_skew_lesao = sqrt((skew_lesao_red)^2 + (skew_lesao_green)^2 + (skew_lesao_blue)^2) / 3;

%         skew_ext_red = skewnesses_exterior(1);
%         skew_ext_green = skewnesses_exterior(2);
%         skew_ext_blue = skewnesses_exterior(3);


        %kurtosis dos canais lesao/exterior
        kurtosis_lesao = kurtosis(double(lesao_segmentada_3_colunas));
%         kurtosis_exterior = kurtosis(double(exterior_segmentado_3_colunas));

        kurt_lesao_red = kurtosis_lesao(1);
        kurt_lesao_green = kurtosis_lesao(2);
        kurt_lesao_blue = kurtosis_lesao(3);
        RGB_kurt_lesao = sqrt((kurt_lesao_red)^2 + (kurt_lesao_green)^2 + (kurt_lesao_blue)^2);

%         kurt_ext_red = kurtosis_exterior(1);
%         kurt_ext_green = kurtosis_exterior(2);
%         kurt_ext_blue = kurtosis_exterior(3);


        %entropia 
        entropy_lesao = entropy(lesao_segmentada);
        entropy_lesao_red = entropy(lesao_segmentada_3_colunas(:,1));
        entropy_lesao_green = entropy(lesao_segmentada_3_colunas(:,2));
        entropy_lesao_blue = entropy(lesao_segmentada_3_colunas(:,3));
        RGB_entropy_lesao = sqrt((entropy_lesao_red)^2 +  (entropy_lesao_green)^2 +  (entropy_lesao_blue)^2);


        %energia
        energy_lesao = sum(sum(lesao_segmentada_3_colunas)) / pixel_imagem; %os zeros nao acrescentam valor


        %color variegation
        cv_lesao_red = sd_lesao_red / max(double(lesao_segmentada_3_colunas(:,1)));
        cv_lesao_green = sd_lesao_green / max(double(lesao_segmentada_3_colunas(:,2)));
        cv_lesao_blue = sd_lesao_blue / max(double(lesao_segmentada_3_colunas(:,3)));
        RGB_cv_lesao = sqrt((cv_lesao_red)^2 + (cv_lesao_green)^2 + (cv_lesao_blue)^2);
        
        features(1,(i-1) * 26 + 1: i * 26) = [red_mean_lesao green_mean_lesao blue_mean_lesao RGB_mean_lesao sd_lesao_red sd_lesao_green ...
            sd_lesao_blue RGB_sd_lesao skew_lesao_red skew_lesao_green skew_lesao_blue RGB_skew_lesao kurt_lesao_red kurt_lesao_green ... 
            kurt_lesao_blue RGB_kurt_lesao entropy_lesao entropy_lesao_red entropy_lesao_green entropy_lesao_blue RGB_entropy_lesao ...
            energy_lesao cv_lesao_red cv_lesao_green cv_lesao_blue RGB_cv_lesao];
    end
    
    
    %Histogram of gradient features
%     featuresHOG = extractHOGFeatures(double(lesao_segmentada_3_colunas));
    
    %RegionProps
    properties = regionprops(mascara_final, 'all');
    %(nota informacao de tamanho nao e muito fiavel pois fotos sao tiradas a diferentes distancias)
    %ainda assim convem normalizar devido as diferentes resolucoes
    Area = max(properties.Area) / pixel_imagem; 
    %max porque ele pode encontrar mais estruturas mas nos queremos a do sinal -> maior area
    MajorAxis = max(properties.MajorAxisLength) / pixel_imagem;
    MinorAxis = max(properties.MinorAxisLength) / pixel_imagem;
    Eccentricity = max(properties.Eccentricity);
    Diameter = max(properties.EquivDiameter) / pixel_imagem;
    Perimeter = max(properties.Perimeter) / pixel_imagem;
    Bounding_box = properties.BoundingBox;
    
    %GLCM
    glcms = graycomatrix(rgb2gray(lesao_segmentada));
    stats = graycoprops(glcms,'all');
    contraste = stats.Contrast;
    correlacao = stats.Correlation;
    energia = stats.Energy;
    homogeneidade = stats.Homogeneity;
    
    
    %Diferencas entre um circulo perfeito com o mesmo raio que a lesao e a
    %lesao
    racio_areas = (pi * (Diameter / 2)^2) / Area;
    racio_perimetros = (2 * pi * (Diameter / 2)) / Perimeter;
    
    %Diferencas entre o quadrado mais pequeno que engloba a lesao e a lesao
    racio_areas_square = (Bounding_box(3) * Bounding_box(4)) / Area;
    racio_perimetros_square = (Bounding_box(3) * 2 + Bounding_box(4) * 2) / Perimeter;
    
    %Features extraídas automáticas
%     lesao_segmentada_gray=rgb2gray(lesao_segmentada);
%     featuresLBP = extractLBPFeatures(lesao_segmentada_gray);
%     featuresHOG = extractHOGFeatures(lesao_segmentada_gray);
    
    features(1,78:92) = [Area MajorAxis MinorAxis Eccentricity Diameter Perimeter contraste correlacao ... 
        energia homogeneidade racio_areas racio_perimetros racio_areas_square racio_perimetros_square];
%     auto_features=[featuresLBP featuresHOG];
end