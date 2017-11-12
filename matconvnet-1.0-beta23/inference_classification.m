function value = inference_classification(im, net)
	im_ = single(im) ; % note: 0-255 range

	% run the CNN
    % To only keep the final result set conserveMemory to true 
    net.conserveMemory = false;
	net.eval({'input', im_});
    
    % obtain the final 90 features
%     features = net.vars(net.getVarIndex('prob')-2).value;
%     features = squeeze(features);

	% obtain the CNN output
	scores = net.vars(net.getVarIndex('prob')).value;
	scores = squeeze(gather(scores));
    	
	[~, best] = max(scores);
    
    value = {best,features};
    
    % Show the classification results
% 	figure(1) ; clf ; imagesc(im);
%     if best == 1
%         title(sprintf('%s (%d), score %.3f', 'benign lesion', best, bestScore));
%     
%     elseif best == 2
%         title(sprintf('%s (%d), score %.3f', 'malignant lesion', best, bestScore));
%     end
end
