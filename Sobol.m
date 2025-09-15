function [S,ST] = Sobol(PopDec,PopObj)
        X = PopDec;
        Y = PopObj;
        [~, d] = size(X); 
        [~, numObjectives] = size(Y); 
        S = zeros(numObjectives,d);
        ST = zeros(numObjectives,d);
     
        for i = 1:d
            Input.Marginals(i).Name = sprintf('X%d', i);
            Input.Marginals(i).Type = 'Uniform';  
            Input.Marginals(i).Parameters = [min(X(:,i)), max(X(:,i))]; 
        end
        myInput = uq_createInput(Input);

        %% Establish separate Kriging models
        for objIdx = 1:numObjectives
            Y_current = Y(:, objIdx); 

            % Create a Kriging model
            MetaOpts.Type = 'Metamodel';
            MetaOpts.MetaType = 'Kriging';
            MetaOpts.ExpDesign.X = X;
            MetaOpts.ExpDesign.Y = Y_current;  % Use only the current objective
            myModel = uq_createModel(MetaOpts);


            %Sobol Sensitivity Analysis
            SobolOpts.Type = 'Sensitivity';
            SobolOpts.Method = 'Sobol';
            SobolOpts.Input = myInput;
            SobolOpts.Model = myModel;
            SobolOpts.Sobol.Order = 1;  
            SobolOpts.Sobol.SampleSize = 1000;  % Sample Count

            SobolAnalysis = uq_createAnalysis(SobolOpts);
            S1 = SobolAnalysis.Results.FirstOrder;
            STotal = SobolAnalysis.Results.Total;
            S(objIdx,:) = S1;
            ST(objIdx,:) = STotal; 
        end
end