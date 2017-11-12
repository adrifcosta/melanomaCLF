% Outubro de 2016
% Base de dados e Análise de Informação
% Mestrado Integrado em Engenharia Biomédica
%
% Adriana Costa
% João Sousa
%
%
% FUNÇAO QUE CONTEM TODAS AS FUNÇOES DE PREPROCESSAMENTO QUE AVALIAMOS;
% PARA ESCOLHER A MELHOR BASTOU AVALIAR A ACCURACY DA MASCARA OBTIDA
% RELATIVAMENTE À MASCARA GROUND TRUTH

function pre_processed  = pre_processing_functions(imagem,mascara_gold)
%   accuracys = zeros(1,14);
    accuracy_final = 0;
    accuracy=0;

%for das cores (2=lab,3=ntsc,4=CIE,5=YCbCr)
% REPRESENTAÇÃO DA IMAGEM NOS DIFERENTES ESPAÇOS DE COR
% for i=1:4
%     if i==2
%        imagem = rgb2lab(imagem) ;
%        
%     elseif i==3
%         imagem = rgb2ntsc(imagem);
%         
%     elseif i==4
%         imagem = rgb2xyz(imagem);
%         
%     elseif i==5
%         imagem = rgb2ycbcr(imagem);
%         
%     elseif i==6
%         imagem = rgb2hsv(imagem);
%     end
%     figure()    

    %CONVERSAO DA IMAGEM PARA ESCALA DE CINZAS
    imagem_gray = rgb2gray(imagem);

    % FUNÇÕES DE THRESHOLD TESTADAS PARA A SEGMENTAÇÃO

    % MÉTODO DE OTSU+ACTIVE CONTOURS

%     level = multithresh(imagem_gray,2);
%     seg_I = imquantize(imagem_gray,level);
%     mascara_otsu = snakes(seg_I,mascara_gold);
%     accuracy_otsu = accuracy_mascara(mascara_otsu,mascara_gold);
%     subplot(1,14,1), imshow(mascara_otsu)
%     accuracys(1,1) = accuracy_otsu;
    
    % MÉTODO ADAPTATIVO
    
%     mascara_adapt = zeros(size(imagem,1),size(imagem,2));
%     accuracy_adapt = 0;
%     for i=0.1:0.2:0.8
%         T = adaptthresh(imagem_gray, i);
%         mascara_adapt_temp = imbinarize(imagem_gray,T);
%         accuracy_adapt_temp = accuracy_mascara(mascara_adapt_temp,mascara_gold);
%         if (accuracy_adapt_temp > accuracy_adapt)
%             accuracy_adapt = accuracy_adapt_temp;
%             mascara_adapt = mascara_adapt_temp;
%         end
%     end
%     subplot(1,14,2), imshow(mascara_adapt)
%     accuracys(1,2) = accuracy_adapt;
    

    % MÉTODO DE OTSU COM RECURSO AO HISTOGRAMA
%     [counts,~] = imhist(imagem_gray,16);
%     T = otsuthresh(counts);
%     mascara_otsuH = imbinarize(imagem_gray,T);
%     accuracy_otsuH = accuracy_mascara(mascara_otsuH,mascara_gold);
%     subplot(1,14,3), imshow(~mascara_otsuH)
%     accuracys(1,3) = accuracy_otsuH;
    
  
    % GRAY THRESHOLD
%     level = graythresh(imagem_gray);
%     mascara_grayT = imbinarize(imagem_gray,level);
%     accuracy_grayT = accuracy_mascara(mascara_grayT,mascara_gold);
%     subplot(1,14,4), imshow(~mascara_grayT)
%     accuracys(1,4) = accuracy_grayT;
  

    % imadjust() AJUSTA OS VALORES DA INTENSIDADE DE COR + ACTIVE CONTOURS
    
%     J1 = imadjust(imagem_gray);
%     mascara_imadjust = snakes(J1,mascara_gold);
%     accuracy_imadjust = accuracy_mascara(mascara_imadjust,mascara_gold);
%     subplot(1,14,5), imshow(mascara_imadjust)
%     accuracys(1,5) = accuracy_imadjust;
    


    % FAZER O SHARP DA IMAGEM RESULTANTE DO IMADJUST()
    
%     J = imsharpen(J1,'Radius',2,'Amount',3);
%     mascara_sharpen = snakes(J,mascara_gold);
%     accuracy_sharpen = accuracy_mascara(mascara_sharpen,mascara_gold);
%     subplot(1,14,6), imshow(mascara_sharpen)
%     accuracys(1,6) = accuracy_sharpen;
    
    % FILTRO LAPLACIANO

%     B = locallapfilt(imagem, 0.1, 0.2);
%     b = rgb2gray(B);
%     mascara_laplace = snakes(b,mascara_gold);
%     accuracy_laplace = accuracy_mascara(mascara_laplace,mascara_gold);
%     subplot(1,14,7), imshow(mascara_laplace)
%     accuracys(1,7) = accuracy_laplace;
   

    % MANIPULAÇAO DAS EDGES
    
%     B = localcontrast(imagem, 0.1, 0.3);
%     b = rgb2gray(B);
%     mascara_edges = snakes(b,mascara_gold);
%     accuracy_edges = accuracy_mascara(mascara_edges,mascara_gold);
%     subplot(1,14,8), imshow(mascara_edges)
%     accuracys(1,8) = accuracy_edges;
    
    % HISTOGRAMA
    
%     J = histeq(imagem_gray);
%     mascara_histeq = snakes(J,mascara_gold);
%     accuracy_histeq = accuracy_mascara(mascara_histeq,mascara_gold);
%     subplot(1,14,9), imshow(mascara_histeq)
%     accuracys(1,9) = accuracy_histeq;
    
    % PROCURAR PONTOS DA MESMA COR
    
%     mascara_min = imextendedmin(imagem_gray,80);
%     accuracy_min = accuracy_mascara(mascara_min,mascara_gold);
%     subplot(1,14,10), imshow(mascara_min)
%     accuracys(1,10) = accuracy_min;
    
    % PROCURAR PONTOS DA MESMA COR
    
%     b = imhmax(imagem_gray,50);
%     mascara_hmax = snakes(b,mascara_gold);
%     accuracy_hmax = accuracy_mascara(mascara_hmax,mascara_gold);
%     subplot(1,14,11), imshow(mascara_hmax)
%     accuracys(1,11) = accuracy_otsu;
    
    % TOP HAT FILTERING
    
%     b = imtophat(imagem_gray,strel('disk',30));
%     mascara_tophat = snakes(b,mascara_gold);
%     accuracy_tophat = accuracy_mascara(mascara_tophat,mascara_gold);
%     subplot(1,14,12), imshow(mascara_tophat)
%     accuracys(1,12) = accuracy_tophat;
    
    
    % REGIONPROPS
    
%     gimg = min(imagem, [], 3 );
%     BW = imbinarize( gimg, .4 ); 
%     st = regionprops( ~BW, 'Area', 'Centroid', 'PixelIdxList' );
%     sel = [st.Area] > numel(BW)*0.025; % at least 2.5% of image size
%     st = st(sel);
%     cntr = .5 * [size(BW,2) size(BW,1)]; % X-Y coordinates and NOT Row/Col
%     d = sqrt( sum( bsxfun(@minus,vertcat( st.Centroid ), cntr ).^2, 2 ) );
%     [~, idx] = min(d);
%     mascara_regionprops = false(size(BW)); 
%     mascara_regionprops( st(idx).PixelIdxList ) = true;
%     accuracy_regionprops = accuracy_mascara(mascara_regionprops,mascara_gold);
%     subplot(1,14,13), imshow(mascara_regionprops)
%     accuracys(1,13) = accuracy_regionprops;
    
    % MARCAR OBJETOS NA IMAGEM
    
    I = rgb2gray(imagem);
    se = strel('disk', 20);
    Io = imopen(I, se);
    Ie = imerode(I, se);
    Iobr = imreconstruct(Ie, I);
    Iobrd = imdilate(Iobr, se);
    Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
    Iobrcbr = imcomplement(Iobrcbr);
    bw = imbinarize(Iobrcbr);
    imagem_bw=imbinarize(I);
    imagem_bw=bwfill(imagem_bw,'holes');
    for i=1:size(imagem_bw,1)
        for j=1:size(imagem_bw,2)
            if(imagem_bw(i,j))==0
                bw(i,j)=1;
            end
        end
    end

    mascara_mark_objects=~bw;
    accuracy_mark_objects = accuracy_mascara(mascara_mark_objects,mascara_gold);
    mascara_final =  mascara_mark_objects;
    %subplot(1,14,14), imshow(mascara_watershed)
    accuracy = accuracy_mark_objects;
%     
%     Se a mascara resultante for muito pouco satisfatoria assume-se a gold
%     standard
    if (accuracy_final < 10)
        mascara_final = mascara_gold;
    end


    pre_processed = {mascara_final accuracy};



end