classdef GSAEA < ALGORITHM
% <multi/many> <real/integer> <expensive>
% Grouping via sensitivity analysis evolutionary algorithm for high-dimensional expensive multi-objective optimization
% k---5

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            k  = Algorithm.ParameterSet(5);

            %% Generate initial population
            NI= 100;
            P     = UniformPoint(NI,Problem.D,'Latin');
            A1    = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));
           %% Sobol sensitivity analysis
            uqlab;
            [S1,~] = Sobol(A1.decs,A1.objs);
           %% Optimization
            while Algorithm.NotTerminated(A1)
                    A2  = Operator(Problem,A1,k,2,S1);
                    A1  = [A1,A2];
            end
        end
    end
end