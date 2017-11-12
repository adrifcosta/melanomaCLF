function imdb = normalizeIMDB(imdb)
	data = imdb.images.data;

	data = bsxfun(@minus, data, imdb.images.data_mean);

	range_min = min(data(:));
	range_max = max(data(:));
	range_multiplier = 127./max(abs(range_min),range_max);
	imdb.images.data = data .* range_multiplier;
end
