function imdb = createIMDB_RAM(folder,dim)   
	% imdb is a matlab struct with several fields, such as:
	%	- images: contains data, labels, ids dataset mean, etc.
	%	- meta: contains meta info useful for statistics and visualization
	imdb = struct();

	% Assuming we have a folder with two
	% subfolders
	positives = dir([folder '/imagens_treino/malignant/*.jpg']);
	negatives = dir([folder '/imagens_treino/benign/*.jpg']);
    imref_aux = imread([folder '/imagens_treino/malignant/', positives(1).name]);
    
    %The size chosen was 256x256 simply because the RAM memory could not
    %save bigger arrays
	imref = imresize(imref_aux, [dim dim]);
	[H, W, CH] = size(imref);

	% Number of images
	NPos = numel(positives);
	NNeg = numel(negatives);
	N = NPos + NNeg;

	% Initializing the structures
	meta.sets = {'train', 'val'};
	meta.classes = {'benign', 'malignant'};

	% Images are in data, in single format
	images.data = zeros(H, W, CH, N, 'single');
	images.data_mean = zeros(H, W, CH, 'single');
	images.labels = zeros(1, N);
	images.set = zeros(1, N);

	numImgsTrain = 0;
	% Loading positive samples
    for i=1:numel(positives)
        
		im = imresize(single(imread([folder '/imagens_treino/malignant/', positives(i).name])), [dim dim]);
		images.data(:,:,:, i) = im;
		images.labels(i) = 2;

		% 90% of images go to training and 10% to validation
		if(numImgsTrain < 0.9 * NPos)
			images.set(i) = 1;
			images.data_mean = images.data_mean + im;
			numImgsTrain = numImgsTrain + 1;
		else
			images.set(i) = 2;
		end
    end
    
	% Loading negative samples
	for i=1:numel(negatives)
        
		im = imresize(single(imread([folder '/imagens_treino/benign/', negatives(i).name])), [dim dim]);
		images.data(:,:,:, NPos+i) = im;
		images.labels(NPos+i) = 1;

		% 90% of images go to training and 10% to validation
		if(numImgsTrain < 0.9 * (NNeg + NPos))
			images.set(NPos+i) = 1;
			images.data_mean = images.data_mean + im;
			numImgsTrain = numImgsTrain + 1;
		else
			images.set(NPos+i) = 2;
		end
    end
    
	images.data_mean = images.data_mean ./ numImgsTrain;

	% Sorting randomly the images 
	indices = randperm(N);
	images.data(:,:,:,:) = images.data(:,:,:,indices);
	images.labels(:) = images.labels(indices);
	images.set(:) = images.set(indices);

	imdb.meta = meta;
	imdb.images = images;
	
end

