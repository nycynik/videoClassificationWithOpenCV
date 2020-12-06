dirDNN = fullfile(mexopencv.root(), 'models');
modelLabels = fullfile(dirDNN, 'synset_words.txt');
modelTxt = fullfile(dirDNN, 'deploy.prototxt');
modelBin = fullfile(dirDNN, 'bvlc_googlenet.caffemodel');  % 51 MB file
files = {modelLabels, modelTxt, modelBin};
urls = {
    'https://cdn.rawgit.com/opencv/opencv/3.4.0/samples/data/dnn/synset_words.txt';
    'https://cdn.rawgit.com/opencv/opencv/3.4.0/samples/data/dnn/bvlc_googlenet.prototxt';
    'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel';
};
if ~isdir(dirDNN), mkdir(dirDNN); end
for i=1:numel(files)
    if exist(files{i}, 'file') ~= 2
        disp('Downloading...')
        urlwrite(urls{i}, files{i});
    end
end