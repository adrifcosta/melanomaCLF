function [net, info] = vgg_train(imdb,expDir)
    run matlab/vl_setupnn ;

	% some common options
	opts.train.batchSize = 100;
	opts.train.numEpochs = 10 ;
	opts.train.continue = true ;
	opts.train.gpus = [] ;
	opts.train.learningRate = [1e-1*ones(1, 10),  1e-2*ones(1, 5)];
	opts.train.weightDecay = 3e-4;
	opts.train.momentum = 0.;
	opts.train.expDir = expDir;
	opts.train.numSubBatches = 1;
	% getBatch options
	bopts.useGpu = numel(opts.train.gpus) >  0 ;
	opts.optMethod = 'gradient'; % ['adagrad', 'gradient']


	% network definition
	net = dagnn.DagNN() ;

	net.addLayer('conv1', dagnn.Conv('size', [11 11 3 64], 'hasBias', true, 'stride', [4, 4], 'pad', [100 100 100 100]), {'input'}, {'conv1'},  {'conv1f'  'conv1b'});
	net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'relu1'}, {});
	net.addLayer('lrn1', dagnn.LRN('param', [5 1 0.0001/5 0.75]), {'relu1'}, {'lrn1'}, {});
	net.addLayer('pool1', dagnn.Pooling('method', 'max', 'poolSize', [3, 3], 'stride', [2 2], 'pad', [0 1 0 1]), {'lrn1'}, {'pool1'}, {});

	net.addLayer('conv2', dagnn.Conv('size', [5 5 64 256], 'hasBias', true, 'stride', [1, 1], 'pad', [2 2 2 2]), {'pool1'}, {'conv2'},  {'conv2f'  'conv2b'});
	net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'relu2'}, {});
	net.addLayer('lrn2', dagnn.LRN('param', [5 1 0.0001/5 0.75]), {'relu2'}, {'lrn2'}, {});
	net.addLayer('pool2', dagnn.Pooling('method', 'max', 'poolSize', [3, 3], 'stride', [2 2], 'pad', [0 0 0 0]), {'lrn2'}, {'pool2'}, {});

	net.addLayer('conv3', dagnn.Conv('size', [3 3 256 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'pool2'}, {'conv3'},  {'conv3f'  'conv3b'});
	net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'relu3'}, {});
	
	net.addLayer('conv4', dagnn.Conv('size', [3 3 256 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu3'}, {'conv4'},  {'conv4f'  'conv4b'});
	net.addLayer('relu4', dagnn.ReLU(), {'conv4'}, {'relu4'}, {});
	
	net.addLayer('conv5', dagnn.Conv('size', [3 3 256 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu4'}, {'conv5'},  {'conv5f'  'conv5b'});
	net.addLayer('relu5', dagnn.ReLU(), {'conv5'}, {'relu5'}, {});
	net.addLayer('pool5', dagnn.Pooling('method', 'max', 'poolSize', [3, 3], 'stride', [2 2], 'pad', [0 0 0 0]), {'relu5'}, {'pool5'}, {});

	net.addLayer('conv6', dagnn.Conv('size', [6 6 256 4096], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool5'}, {'conv6'},  {'conv6f'  'conv6b'});
	net.addLayer('relu6', dagnn.ReLU(), {'conv6'}, {'relu6'}, {});
	net.addLayer('drop6', dagnn.DropOut('rate', 0.5), {'relu6'}, {'drop6'}, {});

	net.addLayer('conv7', dagnn.Conv('size', [1 1 4096 4096], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'drop6'}, {'conv7'},  {'conv7f'  'conv7b'});
	net.addLayer('relu7', dagnn.ReLU(), {'conv7'}, {'relu7'}, {});
	net.addLayer('drop7', dagnn.DropOut('rate', 0.5), {'relu7'}, {'drop7'}, {});
    
    net.addLayer('conv8', dagnn.Conv('size', [1 1 4096 1000], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'drop7'}, {'conv8'},  {'conv8f'  'conv8b'});
    net.addLayer('relu8', dagnn.ReLU(), {'conv8'}, {'relu8'}, {});
	net.addLayer('drop8', dagnn.DropOut('rate', 0.5), {'relu8'}, {'drop8'}, {});
    
    net.addLayer('conv9', dagnn.Conv('size', [1 1 1000 91], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'drop8'}, {'conv9'},  {'conv9f'  'conv9b'});
    net.addLayer('relu9', dagnn.ReLU(), {'conv9'}, {'relu9'}, {});
	net.addLayer('drop9', dagnn.DropOut('rate', 0.5), {'relu9'}, {'drop9'}, {});
    
	net.addLayer('classifier', dagnn.Conv('size', [1 1 91 2], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'drop9'}, {'classifier'},  {'conv10f'  'conv10b'});
	net.addLayer('prob', dagnn.SoftMax(), {'classifier'}, {'prob'}, {});
	net.addLayer('objective', dagnn.Loss('loss', 'log'), {'prob', 'label'}, {'objective'}, {});
	net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'prob','label'}, 'error') ;
	% -- end of the network

	% initialization of the weights
	initNet(net, 1/100, opts.optMethod);
    
	% training
	info = cnn_train_dag(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train, 'val', find(imdb.images.set == 2)) ;
end

function initNet(net, f, optMethod)
	net.initParams();
    
	f_ind = net.layers(1).paramIndexes(1);
	b_ind = net.layers(1).paramIndexes(2);
	net.params(f_ind).value = 10*f*randn(size(net.params(f_ind).value), 'single');
	net.params(f_ind).learningRate = 1;
	net.params(f_ind).weightDecay = 1;
	net.params(f_ind).trainMethod = optMethod;
	net.params(b_ind).trainMethod = optMethod;

	for l=2:length(net.layers)
		if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);

			[h,w,in,out] = size(net.params(f_ind).value);
			net.params(f_ind).value = f*randn(size(net.params(f_ind).value), 'single');
			net.params(f_ind).learningRate = 1;
			net.params(f_ind).weightDecay = 1;
			net.params(f_ind).trainMethod = optMethod;

			net.params(b_ind).value = f*randn(size(net.params(b_ind).value), 'single');
			net.params(b_ind).learningRate = 0.5;
			net.params(b_ind).weightDecay = 1;
			net.params(b_ind).trainMethod = optMethod;
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
