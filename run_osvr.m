%% example of training and testing OSVR for expression intensity estimation
clear all; close all;

%% load data
% train_data_seq: an array of cells containing training feature sequences,
% each cell contains a D*T matrix where D is dimension of feature and T is
% the sequence length
% train_label_seq: an array of cells containing training intensity labels
% for all the sequences, each cell contains a K*2 matrix where K is the
% number of frames with labeled intensities. The first column is the index
% of frame and the second column is associated intensity value
% test_data: a D*T' matrix containing testing frames, where D is the
% dimension of feature and T' is number of testing frames

features = {'lbp', 'hog', 'landmarks', 'sift', 'gabor', 'dct'};
emotions = {'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'};
%% features = {'landmarks'};
%% emotions = {'Fear'};

base_dir = '/home/liuliang/DISK_2T/datasets/movie/new/mat_feature/labels_2142';
fid=fopen('log_osvr_l1.txt','w');

for i = 1:length(emotions)
    for j = 1:length(features)
        pccs = zeros(5, 1);
        iccs = zeros(5, 1);
        maes = zeros(5, 1);
        for k_fold_index = 0:4
            emotion = emotions{i};
            fprintf('For %s - %s - fold_%d:\n', emotion, features{j}, k_fold_index);
            fprintf(fid, 'For %s - %s - fold_%d:\n', emotion, features{j}, k_fold_index);

            dataset = sprintf('%s/%s_%s_fold%d.mat', base_dir, features{j}, emotion, k_fold_index);
            %% dataset = 'data.mat';
            load(dataset,'train_data_seq','train_label_seq','test_data','test_label');
            
            %% define constant
            loss = 1; % loss function of OSVR
            bias = 1; % include bias term or not in OSVR
            lambda = 1; % scaling parameter for primal variables in OSVR
            gamma = [100 1]; % loss balance parameter
            smooth = 1; % temporal smoothness on ordinal constraints
            epsilon = [0.1 1]; % parameter in epsilon-SVR
            rho = 0.1; % augmented Lagrangian multiplier
            flag = 0; % unsupervise learning flag
            max_iter = 300; % maximum number of iteration in optimizating OSVR
            
            %% Training 
            % formalize coefficients data structure
            fprintf('Construct params...\n');
            [A,c,D,nInts,nPairs,weight] = constructParams(train_data_seq,train_label_seq,epsilon,bias,flag);
            fprintf('Num of pairs: %d\n', nPairs);
            fprintf(fid, 'Num of pairs: %d\n', nPairs);
            %% continue;

            mu = gamma(1)*ones(nInts+nPairs,1); % change the values if you want to assign different weights to different samples
            mu(nInts+1:end) = gamma(2)/gamma(1)*mu(nInts+1:end);
            if smooth % add temporal smoothness
                mu = mu.*weight;
            end
            % solve the OSVR optimization problem in ADMM
            fprintf('Solver admm...\n');
            [model,history,z] = admm(A,c,lambda,mu,'option',loss,'rho',rho,'max_iter',max_iter,'bias',1-bias); % 
            theta = model.w;
                
            %% Testing 
            % perform testing
            fprintf('Test...\n');
            dec_values =theta'*[test_data; ones(1,size(test_data,2))];
            % compute evaluation metrics
            RR = corrcoef(dec_values,test_label);  
            ee = dec_values - test_label; 
            dat = [dec_values; test_label]'; 
            ry_test = RR(1,2); % Pearson Correlation Coefficient (PCC)
            abs_test = sum(abs(ee))/length(ee); % Mean Absolute Error (MAE)
            mse_test = ee(:)'*ee(:)/length(ee); % Mean Square Error (MSE)
            icc_test = ICC(3,'single',dat); % Intra-Class Correlation (ICC)

            fprintf('PCC: %.5f\n', ry_test);
            fprintf('ICC: %.5f\n', icc_test);
            fprintf('MAE: %.5f\n', abs_test);
            fprintf(fid, 'PCC: %.5f\n', ry_test);
            fprintf(fid, 'ICC: %.5f\n', icc_test);
            fprintf(fid, 'MAE: %.5f\n', abs_test);
            
            pccs(k_fold_index + 1) = ry_test;
            iccs(k_fold_index + 1) = icc_test;
            maes(k_fold_index + 1) = abs_test;
            %% Visualize results
            %% plot(test_label); hold on; 
            %% plot(dec_values,'r');
            %% legend('Ground truth','Prediction')
        end
        %% cal mean metric
        fprintf('Result of %s - %s:\n', emotion, features{j});
        fprintf('++ PCC: %.5f\n', mean(pccs));
        fprintf('++ ICC: %.5f\n', mean(iccs));
        fprintf('++ MAE: %.5f\n', mean(maes));
        fprintf(fid, 'Result of %s - %s:\n', emotion, features{j});
        fprintf(fid, '++ PCC: %.5f\n', mean(pccs));
        fprintf(fid, '++ ICC: %.5f\n', mean(iccs));
        fprintf(fid, '++ MAE: %.5f\n', mean(maes));
    end
end
fclose(fid);
