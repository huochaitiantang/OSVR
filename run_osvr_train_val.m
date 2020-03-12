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

%% features = {'lbp', 'hog', 'landmarks', 'sift', 'gabor', 'dct'};
%% features = {'lbp_pca_0.95', 'cat_lbp_pca_0.95_landmarks'};
%% features with 1000-d will cost all the memory
features = {'landmarks'};
emotions = {'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'};
%%emotions = {'Angry', 'Disgust', 'Fear', 'Sad', 'Surprise'};
%% features = {'landmarks'};
%% emotions = {'Happy'};

base_dir = '/home/liuliang/DISK_2T/datasets/SFEID/mid/mat_feature';
base_sample_dir = base_dir;
% base_sample_dir = sprintf('%s/labels_2142', base_dir);
base_test_clips_dir = sprintf('%s/test_clips', base_dir);

loss = 2; % loss function of OSVR
fid=fopen('log_osvr_l2.txt','a');

for i = 1:length(emotions)
    for j = 1:length(features)
        pccs = 0;
        iccs = 0;
        maes = 0;

        fname = sprintf('res/res_osvr_l%d_%s_%s.txt', loss, features{j}, emotions{i});
        fres = fopen(fname, 'w');
        
        emotion = emotions{i};
        test_clip = sprintf('%s/%s_test.txt', base_test_clips_dir, emotion);
        [clip_ids, frame_cnts]=textread(test_clip, '%s %d');
        clip_cnt = length(clip_ids);

        %for mm = 1:length(clip_ids)
        %    fprintf('clip_id:%s frame_cnt:%d\n', clip_ids{mm}, frame_cnts(mm));
        %end

        fprintf('For %s - %s:\n', emotion, features{j});
        fprintf(fid, 'For %s - %s:\n', emotion, features{j});

        dataset = sprintf('%s/%s_%s.mat', base_sample_dir, features{j}, emotion);
        %% dataset = 'data.mat';
        load(dataset,'train_data_seq','train_label_seq','test_data','test_label');
        
        %% define constant
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
        %%continue;

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
        
        % write the predict result
        base = 0;
        for idx = 1:length(clip_ids)
            fprintf(fres, '%s', clip_ids{idx});
            for offset = 1:frame_cnts(idx)
                fprintf(fres, ' %.5f', dec_values(base + offset));
            end
            fprintf(fres, '\n');

            % calculate metric based on clip
            pred = dec_values(base+1: base+frame_cnts(idx));
            gt = test_label(base+1: base+frame_cnts(idx));
            % compute evaluation metrics
            RR = corrcoef(pred, gt);  
            ee = pred - gt; 
            dat = [gt; pred]'; 
            ry_test = RR(1,2); % Pearson Correlation Coefficient (PCC)
            abs_test = sum(abs(ee))/length(ee); % Mean Absolute Error (MAE)
            mse_test = ee(:)'*ee(:)/length(ee); % Mean Square Error (MSE)
            icc_test = ICC(3,'single',dat); % Intra-Class Correlation (ICC)

            pccs = pccs + ry_test;
            iccs = iccs + icc_test;
            maes = maes + abs_test;

            base = base + frame_cnts(idx);
        end

        pccs = pccs / clip_cnt;
        iccs = iccs / clip_cnt;
        maes = maes / clip_cnt;
            

            % fprintf('PCC: %.5f\n', ry_test);
            % fprintf('ICC: %.5f\n', icc_test);
            % fprintf('MAE: %.5f\n', abs_test);
            % fprintf(fid, 'PCC: %.5f\n', ry_test);
            % fprintf(fid, 'ICC: %.5f\n', icc_test);
            % fprintf(fid, 'MAE: %.5f\n', abs_test);
            % 
            % pccs(k_fold_index + 1) = ry_test;
            % iccs(k_fold_index + 1) = icc_test;
            % maes(k_fold_index + 1) = abs_test;
            %% Visualize results
            %% plot(test_label); hold on; 
            %% plot(dec_values,'r');
            %% legend('Ground truth','Prediction')
        fclose(fres);
        %% cal mean metric
        fprintf('Result of %s - %s:\n', emotion, features{j});
        fprintf('++ PCC: %.5f\n', pccs);
        fprintf('++ ICC: %.5f\n', iccs);
        fprintf('++ MAE: %.5f\n', maes);
        fprintf(fid, 'Result of %s - %s:\n', emotion, features{j});
        fprintf(fid, '++ PCC: %.5f\n', pccs);
        fprintf(fid, '++ ICC: %.5f\n', iccs);
        fprintf(fid, '++ MAE: %.5f\n', maes);
    end
end
fclose(fid);
