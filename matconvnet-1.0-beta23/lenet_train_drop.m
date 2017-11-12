function [net, info] = lenet_train_drop(imdb,expDir)
    run matlab/vl_setupnn ;

	% some common options
	opts.train.batchSize = 100 ;
	opts.train.numEpochs = 15 ;
	opts.train.continue = true ;
	opts.train.gpus = [] ;
	opts.train.learningRate = 0.001 ;
	opts.train.expDir = expDir;
	opts.train.numSubBatches = 1 ;
	% getBatch options
	bopts.useGpu = numel(opts.train.gpus) >  0 ;


	% network definition
	net = dagnn.DagNN() ;
	net.addLayer('conv1', dagnn.Conv('size', [5 5 3 20], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'input'}, {'conv1'},  {'conv1f'  'conv1b'});
	net.addLayer('pool1', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 0 0 0]), {'conv1'}, {'pool1'}, {});

	net.addLayer('conv2', dagnn.Conv('size', [5 5 20 50], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool1'}, {'conv2'},  {'conv2f'  'conv2b'});
	net.addLayer('pool2', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 0 0 0]), {'conv2'}, {'pool2'}, {});
	net.addLayer('drop2', dagnn.DropOut('rate', 0.7), {'pool2'}, {'drop2'}, {});

	net.addLayer('conv3', dagnn.Conv('size', [4 4 50 500], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'drop2'}, {'conv3'},  {'conv3f'  'conv3b'});
	net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'relu3'}, {});
	net.addLayer('drop3', dagnn.DropOut('rate', 0.7), {'relu3'}, {'drop3'}, {});
    
    net.addLayer('conv4', dagnn.Conv('size', [1 1 500 92], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'drop3'}, {'conv4'},  {'conv4f'  'conv4b'});
	net.addLayer('relu4', dagnn.ReLU(), {'conv4'}, {'relu4'}, {});
	net.addLayer('drop4', dagnn.DropOut('rate', 0.7), {'relu4'}, {'drop4'}, {});

	net.addLayer('classifier', dagnn.Conv('size', [1 1 92 2], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'drop4'}, {'classifier'},  {'conv5f'  'conv5b'});
	net.addLayer('prob', dagnn.SoftMax(), {'classifier'}, {'prob'}, {});
	net.addLayer('objective', dagnn.Loss('loss', 'log'), {'prob', 'label'}, {'objective'}, {});
	net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'prob','label'}, 'error') ;
	% -- end of the network
   
	% initialization of the weights
	initNet_xavier(net);
    
	%training
	info = cnn_train_dag(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train, 'val', find(imdb.images.set == 2)) ;
end

function initNet_xavier(net)
	net.initParams();
	for l=1:length(net.layers)
		% is a convolution layer?
		if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);

			[h,w,in,out] = size(net.params(f_ind).value);
			xavier_gain = 0.5*sqrt(2/(h*w*out));
			net.params(f_ind).value = xavier_gain*randn(size(net.params(f_ind).value), 'single');
			net.params(f_ind).learningRate = 1;
			net.params(f_ind).weightDecay = 1;

			net.params(b_ind).value = zeros(size(net.params(b_ind).value), 'single');
			net.params(b_ind).learningRate = 0.5;
			net.params(b_ind).weightDecay = 1;
		end
	end
end

% function on charge of creating a batch of images + labels
function inputs = getBatch(opts, imdb, batch)
	images = imdb.images.data(:,:,:,batch) ;
	labels = imdb.images.labels(1,batch) ;
	if opts.useGpu > 0
  		images = gpuArray(images) ;
	end

	inputs = {'input', images, 'label', labels} ;
end
