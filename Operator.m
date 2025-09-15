function Offspring = Operator(Problem,POP,k,Lp,S1)
    %  Initialization
    PopDec = POP.decs;
    PopObj = POP.objs;
    [~,M] = size(PopObj);
    [~,D] = size(PopDec);
    %% k-means cluster
    NM = M;
    [idx, Clusterc] = kmeans(PopDec, NM);
    
    %% Build RBF Objcitve model based on cluster
    RBF_para = cell(1,NM);
    for i = 1: NM
        Objs_Surrogate = PopObj(idx==i, :);
        Decs_Surrogate = PopDec(idx==i, :);
        for j = 1:Problem.M
            RBF_para{i}(j) = RBFCreate(Decs_Surrogate, Objs_Surrogate(:,j), 'cubic');
        end
    end
    
    %% Build RBF Con model
    Con = calCon(PopObj);
    RBF_Con = RBFCreate(PopDec, Con, 'cubic');
    
    %% Build RBF SDE model
    SDE = CalSde(PopObj);
    RBF_SDE = RBFCreate(PopDec, SDE, 'cubic');
    
     %% Two Group
        S1_s = sum(S1, 1);
        mean_value = mean(S1_s);
        Sub{1} = find(S1_s <= mean_value);  %Convergence
        Sub{2} = find(S1_s > mean_value);   %Diversity

      
      %% Optimization
       % Convergence
       [~,idx1] =  sort(Con);
       PopDec1 = PopDec(idx1(1:Problem.N),:);
       Con1 = Con(idx1(1:Problem.N),:);
       for gen = 1: 10
           OffDec = PopDec1;
           N1 = size(PopDec1,1);
           MatingPool = TournamentSelection(2,2*N1,Con1);
           NewDec = OperatorDE1(Problem,PopDec1,PopDec1(MatingPool(1:end/2),:),PopDec1(MatingPool(end/2+1:end),:),{1,0.5,D/length(Sub{1})/2,20});
           OffDec(:,Sub{1}) = NewDec(:,Sub{1});
           newCon = RBF_Predictor_1(OffDec, RBF_Con, 1);
           updated = Con1 > newCon;
           PopDec1(updated,:) = OffDec(updated,:);
           Con1(updated)      = newCon(updated);
       end
       
       % Diversity 
       [~, idx2] = mink(Con, Problem.N);
       PopObj2 = PopObj(idx2,:);
       PopDec2 = PopDec(idx2,:);
       Con2 = Con(idx2,:);
       for gen = 1: M
           N2 = size(PopDec2,1);
           OffObj = zeros(N2, M);
           OffDec       = PopDec2(TournamentSelection(2,N2,Con2),:);
           MatingPool = TournamentSelection(2,N2,Con2);
           NewDec       = OperatorGA1(Problem,PopDec2(MatingPool,:));
           OffDec(:, Sub{2}) = NewDec(:, Sub{2});
           idx   = assignClusters(OffDec, Clusterc,Lp);   
           for j = 1: NM   
               present = find(idx == j);
               OffObj(present,:) = RBF_Predictor(OffDec(present,:), RBF_para{j}, M);   
           end
           ComObj = [PopObj2; OffObj];
           ComDec = [PopDec2; OffDec];
           idx2 = DSelectNew(ComObj,Problem.N);
           PopObj2 = ComObj(idx2,:);
           PopDec2 = ComDec(idx2,:);
           Con2   = RBF_Predictor_1(PopDec2, RBF_Con, 1);
       end
       
       %% Merge
       OffDec(:,Sub{1}) = PopDec1(:,Sub{1});
       OffDec(:,Sub{2}) = PopDec2(:,Sub{2});
       Repeated = ismember(OffDec, PopDec, 'rows');
       OffDec = OffDec(~Repeated,:);
       newSDE = RBF_Predictor_1(OffDec , RBF_SDE, 1);
       [~,idx3] = mink(-newSDE,k);
        Offspring = Problem.Evaluation(OffDec(idx3,:));

end

function Con = calCon(PopObj)
     % Calculate the convergence of each solution
    [z,znad]      = deal(min(PopObj),max(PopObj));
%     [PopObj,z,~] = Normalization(PopObj,z,znad);
     Con =  sqrt(sum((PopObj-z).^2,2));
end

function Fitness = CalSde(PopObj)
    % Calculate the fitness by shift-based density
    N      = size(PopObj,1);
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
    Dis    = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    Fitness = min(Dis,[],2);
end