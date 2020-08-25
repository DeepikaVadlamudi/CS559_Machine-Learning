%{
Author - Sesha Phani Deepika Vadlamudi
Project - CS 559 Classification using Adaboost with MLE as the weak
clasifier
Version 2
Notes - Adding comments
%}
dataset = readmatrix('pima-indians-diabetes')
[m,~]=size(dataset);
%Splitting data into Training data and Testing data
P = 0.66 ;
idx = randperm(m) ;
Trainingset = dataset(idx(1:round(P*m)),:);
Testingset = dataset(idx(round(P*m)+1:end),:);
% All the observations of training data are assigned equal weight
D = repmat((1/length(Trainingset)), [length(Trainingset), 1])
D0 = D(Trainingset(:,9)==0);
D1 = D(Trainingset(:,9)==1);
%for the purpose of implementation and understanding we have used only one
%feature
active_feat = 2;
%--------Training--------%
%M denotes the Number of base classifiers chosen
M=5;
for k=1:M
 % Initially each data point (observation) is assumed to be correctly classified by the weak classifier
 data_1 = [Trainingset, repmat(0,length(Trainingset),1)];
 TrainingData0 = transpose(Trainingset(Trainingset(:,9)==0,active_feat));
 TrainingData1 = transpose(Trainingset(Trainingset(:,9)==1,active_feat));
 % Estimate parameters of the distributions of each class corresponding to kth weak classfier
 mean1(k)= (TrainingData0*D0)/sum(D0) ;
 mean2(k) = (TrainingData1*D1)/sum(D1) ;
 var1(k) = var(Trainingset(Trainingset(:,9)==0,active_feat),D0) ;
 var2(k) = var(Trainingset(Trainingset(:,9)==1,active_feat),D1) ;
 % Computation of prior probabilities of the data based on the weights assigned to the data points
% These prior probabilities correspond to the kth classifier
 prior1tmp = sum(D0);
 prior2tmp = sum(D1);
 prior1(k) = prior1tmp/(prior1tmp+prior2tmp) ;
 prior2(k) = prior2tmp/(prior1tmp+prior2tmp);
 correct=0;
 wrong=0;
 % Computation of likelihood estimates and hence posterior probabilities of all the training examples
 % Also classify the training examples using the kth weak classifier based on the posterior probabilities
 invvar1(k) = inv(var1(k));
 invvar2(k) = inv(var2(k));
 for i = 1:length(Trainingset)
 lklhood1 = exp(-(Trainingset(i,active_feat)-mean1(k))^2/(2*var1(k))) /sqrt(var1(k));
 lklhood2 = exp(-(Trainingset(i,active_feat)-mean2(k))^2/(2*var2(k)))/sqrt(var2(k));
 post1 = lklhood1*prior1(k);
 post2 = lklhood2*prior2(k);
 % Determine the data points which are misclassified by the kth weak classifier
 if(post1>post2 && Trainingset(i,9)==0)
 correct = correct+1;
 data_1(i,10)=1;
 elseif(post1 < post2 && Trainingset(i,9)==1)
 correct = correct+1;
 data_1(i,10)=1;
 else
 wrong = wrong +1;
 end
 end

% Modify the weights assigned to the data points based on whether the data point is correctly classified
% or not. Also determine the weight of the kth weak classifier while aggregating the results of all the
% weak classifier to have a strong classifier
%% Epsilon Calculation
 Eps_num = 0;
 Eps_den = 0;
 for i =1:length(Trainingset)
 Eps_den = Eps_den+D(i);
 if(data_1(i,10)==0)
 Eps_num = Eps_num +D(i);
 end
 end
 Eps = Eps_num/Eps_den;
 %% Alpha Calculation
 if(Eps>0&&Eps<1)
 alpha(k)= 0.5*log((1-Eps)/Eps);%% In fact, it should be assigned largest number. Here, we consider 2 to be
largest
 elseif(Eps==1)
 alpha(k) = -2;%% In fact, it should be assigned smallest number. Here, we consider -2 to be smallest
 end
 %% Computation of Weight distribution
 for i = 1:length(Trainingset)
 if(data_1(i,10)==1)
 D(i) = D(i)*exp(-alpha(k));
 else
 D(i)=D(i)*exp(alpha(k));
 end
 end
 z = sum(D);
 D = D/z;
 % New weights of data points of class “0”
 D0 = D(Trainingset(:,9)==0);
 % New weights of data points of class “1”
 D1 = D(Trainingset(:,9)==1);
end
%-----Testing-----%
correct = 0;
wrong = 0;
% Classify each data point in the test data
for i = 1:length(Testingset)
 post1 =0;
 post2 = 0;
 % Classify the ith data point in the test data by all the M weak classifiers, aggregate these decisions with
 % the corresponding weights, and come up with the final classification of this data point.
 for k = 1:M
 lklhood1 = exp(-(Testingset(i,active_feat)-mean1(k))^2/(2*var1(k)))/sqrt(var1(k))
 lklhood2 = exp(-(Testingset(i,active_feat)-mean2(k))^2/(2*var2(k)))/sqrt(var2(k))
 post1 = post1 +alpha(k)*lklhood1*prior1(k);
 post2 = post2 +alpha(k)*lklhood2*prior2(k);
 end
 if(post1>post2 && Testingset(i,9)==0)
 correct = correct+1;
 elseif(post1<post2&&Testingset(i,9)==1)
 correct = correct+1;
 else
 wrong = wrong+1;
 end
end
