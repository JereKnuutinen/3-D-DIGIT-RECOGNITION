function C = digit_classify(testdata)
%%%%%% C = digit_classify(testdata) is integer that corresponds to
%%%%%% recognized digit.
%%%%%% digit classify function classifies a single 3xN digit.
%%%%%% data is first preprocessed.
%%%%%% featues are extracted by using SVD on mean subtracted data.
%%%%%% classification is performed by using Bayesian classifier.


% some constants
m = 3; % number of colums in each stroke file.
n = 100; %number of rows in each loaded stroke (after interpolating).
N = 100; %number of training samples (one class).
num_of_classes = 10; %number of classes in total.
r = 10; %number of columns of U used. this is needed in when features are extracted.

%%% When you want to clasifie a single unknow sample 
[X,avg] = Load_TrainData(num_of_classes,N,n,m); %this function loads raw training data and makes part of data preprocessing. 
[traindata,trainclass,U] = Features_TrainData(X,avg,r,N,num_of_classes); %this function extracts features from the traindata. 
[testingData] = Features_TestData(testdata,n,m,r,U,avg);%this function extract features from the single test data sample.
[C1, Pest, muEst, SigmaEst] = bayes3(trainclass,traindata,testingData); %classification is performed in bayes3 function.
C = C1-1; % Labels are 1-10 so if we want to get 0-9, we have to subtract 1 in order to get 0-9.

end

function [X,avg_digit] = Load_TrainData(num_of_classes,N,n,m)    
%%% [X,avg_digit] = Load_TrainData(num_of_classes,N,n,m) returns X and
%%% average digit. X is data matrix that contains reshaped digits.
%%% avg_digit is a average digit.
%%% this function loads training data. 
%%% also part of the data preprocessing happens here.
X = []; %matrix that contains all digits in vector form. 
% next load training data into matlab.
for i = 1:num_of_classes
    for j = 1:N
        % load data to matlan
        stroke = ['trainingData/stroke_',num2str(i-1,'%2d'),'_0',num2str(j,'%03d'),'.mat']; % file name
        Data = load(stroke); %load spesific digit
        
        %separate data into x,y,z, coordinates.
        datax = [Data.pos(:,1)]; %x coordinate
        datay = [Data.pos(:,2)]; %y coordinate
        dataz = [Data.pos(:,3)]; %z coordinate
        
        % resample data that every sample has same number of rows
        datax1 = interp1(linspace(min(datax),max(datax),size(datax,1)),datax,linspace(min(datax),max(datax),n)); %x-coordinates
        datay1 = interp1(linspace(min(datay),max(datay),size(datay,1)),datay,linspace(min(datay),max(datay),n)); %y-coordinates
        dataz1 = interp1(linspace(min(dataz),max(dataz),size(dataz,1)),dataz,linspace(min(dataz),max(dataz),n)); %z-coordinates

        %center digit
        datax_c = datax1 - mean(datax1); %center x-coordinates
        datay_c = datay1 - mean(datay1); %center y-coordinates
        dataz_c = dataz1 - mean(dataz1); %center z-coordinates
        
        temp = [datax_c' datay_c' dataz_c']; %temporary matrix to intepolated and centered digit
        temp_res = temp(:);  %reshape  digit into vector
        
        %put reshaped digits into a matrix X
        X = [X, temp_res]; %create matrix X that contains all interpolated and reshaped digits
    end
end

%average digit
avg_digit = mean(X,2);

end

function [traindata,trainclass,U] = Features_TrainData(X,average,r,N,num_of_classes)
%%% [traindata,trainclass,U] = Features_TrainData(X,avg,r,N,num_of_classes)
%%% returns traindata matrix that contains extracted features. In addition
%%% function returns class labels (trainclass) and matrix U.
%%% this function subtracts mean from every column of X and calculates svd
%%% on mean subtracted training data.

% subtracting average form every column of X
% because we perform svd on mean subtracted data
for i = 1:num_of_classes*N
    X(:,i) = X(:,i) - average; %here we subtract avg digit from every column of X
end

%%%SVD%%%
[U,S,V] = svd(X); %calculate svd 

%%%% notice: use these lines  if svd function is too advanced function for
%%%% this practical assigment. few lines of code below also determines basicly U matrix
%%%%(X*X^T*U=U*sigma^2)
% [U,V] = eig(X*X');
% %U = U;
% U = flip(U,2);

% training data feature extraction
% inner product between column vectors of X and U.
for j = 1:num_of_classes*N
     x(:,j) = X(:,j)'*U(:,1:r); % x is our data 
end
traindata = x;

%class labeling
for i = 1:num_of_classes
    trainclass(((i-1)*N+1):i*N) = i; %class labels are 1-10
end

end

function [testingData] = Features_TestData(testdata,n,m,r,U,average)
%%% [testingData] = Features_TestData(testdata,n,m,r,U,avg) returns
%%% test data.
%%% this function preprocess a single testdata sample
%%% this function extract features from the single test sample

% load test data and make feature extraction
% separate test data into x,y and z coordinates

% check if input testdata is stuct or not
condition = isstruct(testdata);
if condition == 1
    testdatax = [testdata.pos(:,1)]; %x coordinate
    testdatay = [testdata.pos(:,2)]; %y coordinate
    testdataz = [testdata.pos(:,3)]; %z coordinate 
else
    testdatax = [testdata(:,1)];%x coordinate
    testdatay = [testdata(:,2)];%y coordinate
    testdataz = [testdata(:,3)];%z coordinate
end


%resample data that every sample has same number of rows
testdatax = interp1(linspace(min(testdatax),max(testdatax),size(testdatax,1)),testdatax,linspace(min(testdatax),max(testdatax),n)); %x coordinate
testdatay = interp1(linspace(min(testdatay),max(testdatay),size(testdatay,1)),testdatay,linspace(min(testdatay),max(testdatay),n)); %y coordinate
testdataz = interp1(linspace(min(testdataz),max(testdataz),size(testdataz,1)),testdataz,linspace(min(testdataz),max(testdataz),n)); %z coordinate

% center test digit
testdatax_c = testdatax - mean(testdatax); %x coordinate
testdatay_c = testdatay - mean(testdatay); %y coordinate
testdataz_c = testdataz - mean(testdataz); %z coordinate

test_temp = [testdatax_c' testdatay_c' testdataz_c']; % temporary matrix for a digit
test_temp_res = test_temp(:); %reshape test digit into vector
test_temp_res = test_temp_res - average; %subtract the average digit

testingData = (test_temp_res'*U(:,1:r))'; %feature extraction for test digit

end

function [C ,P_prior,mu_EST,sigma_EST] = bayes3(trainclass,traindata,data)
    %%%% [C ,P_prior,mu_EST,sigma_EST] = bayes3(trainclass,traindata,data)
    %%%% returns class label C
    %%% this function performs classification
    %%% bayesian classifier is used
    c = max(trainclass); %number of trainnig classes
    nTest = size(data,2); %number of samples in the testdata
    nTrain = length(traindata);   % number of samples in the traindata
    mu_EST = zeros(size(traindata,1),c); %allocate memory for estimated means
    sigma_EST = zeros(size(traindata,1),size(traindata,1),c); %allocate memory for estimated covarinces
    P_prior = zeros(1,3); %allocate memory for priors
    g = zeros(c,nTest); %allocate memory for discriminat values

    for ii=1:c
        classi = traindata(:,find(trainclass == ii)); %find spesific class
        mu_EST(:,ii) = mean(classi,2); %estimate mean of each class
        sigma_EST(:,:,ii) = cov(classi'); %estimate covariance
        P_prior(ii) = length(classi)/nTrain; %estimate prior probability
        for jj = 1:nTest
            % element (ii,jj) corresponds iith discriminant function with
            % sample jj
            g(ii,jj) = -0.5*(data(:,jj)-mu_EST(:,ii))'*inv(sigma_EST(:,:,ii))*(data(:,jj)-mu_EST(:,ii)) - 0.5*log(det(sigma_EST(:,:,ii))) + log(P_prior(ii));
        end
        
    end
    [~,C] = max(g); %choose maximal class
end