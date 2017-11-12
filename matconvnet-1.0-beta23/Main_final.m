%Necessary to run at least the first time
%run ./matlab/vl_compilenn;
run ./matlab/vl_setupnn;

%Load testing images' label
vetor_gt = load('imagens_treino/vetor_gt');
vetor_gt = vetor_gt.vetor;

%Perform training 
imdb = createIMDB_RAM(pwd,58);
imdb = normalizeIMDB(imdb);
netPre = load('nets/imagenet-matconvnet-alex.mat');
[net, info] = alexnet_train_finetuning(imdb,'results/exp_binary_04',netPre); %58
final_values = test_cnn(net,vetor_gt,58);
accuracy_benigno1 = final_values{1};
accuracy_maligno1 = final_values{2};
jaccard_index1 = final_values{3};

imdb = createIMDB_RAM(pwd,58);
[net, info4] = alexnet_train_bnorm(imdb,'results/output_alexbnorm_e1'); %58
final_values = test_cnn(net,vetor_gt,58);
accuracy_benigno4 = final_values{1};
accuracy_maligno4 = final_values{2};
jaccard_index4 = final_values{3};

imdb = createIMDB_RAM(pwd,31);
imdb = normalizeIMDB(imdb);
[net, info3] = lenet_train_drop(imdb,'results/output_lenet_e1'); %31
final_values = test_cnn(net,vetor_gt,31);
accuracy_benigno2 = final_values{1};
accuracy_maligno2 = final_values{2};
jaccard_index2 = final_values{3};

imdb = createIMDB_RAM(pwd,54);
imdb = normalizeIMDB(imdb);
[net, info2] = vgg_train(imdb,'results/output_vggf_e1'); %54
final_values = test_cnn(net,vetor_gt,54);
accuracy_benigno3 = final_values{1};
accuracy_maligno3 = final_values{2};
jaccard_index3 = final_values{3};
