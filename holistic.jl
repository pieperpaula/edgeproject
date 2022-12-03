using CSV, LinearAlgebra, Random, Gurobi, JuMP, Statistics, Combinatorics, DataFrames, StatsBase

data = select!(CSV.read("/home/gbonheur/edge/game_difference.csv", DataFrame), Not(:Column1));
data_reduced = select!(data, Not([:ROUND, :WTEAM, :WSCORE, :LTEAM, :LSCORE, :confW, :confL, :BARTHAG]));

# convert FTRD column to numeric
#data_reduced.FTRD = parse.(Float64, data_reduced.FTRD);

# create train test split with test being years 2019 and 2021
train = filter(row -> row.YEAR != 2019 && row.YEAR != 2021, data_reduced);
valid = filter(row -> row.YEAR == 2019, data_reduced);
test = filter(row -> row.YEAR == 2019 || row.YEAR == 2021, data_reduced);

# drop year column
train = select!(train, Not(:YEAR));
valid = select!(valid, Not(:YEAR));
test = select!(test, Not(:YEAR));

# create X and y matrices with y being SCORE_DIFF
X_train = Matrix(train[:, Not(:SCORE_DIFF)]);
y_train = train[:, :SCORE_DIFF];
X_valid = Matrix(valid[:, Not(:SCORE_DIFF)]);
y_valid = valid[:, :SCORE_DIFF];
X_test = Matrix(test[:, Not(:SCORE_DIFF)]);
y_test = test[:, :SCORE_DIFF];

function compute_mse(X, y, beta)
    n,p = size(X)
    return sum((X*beta .- y).^2)/n
end

function get_transformation(X, e)
    p = size(X)[2] * 4
    X_new = zeros(size(X)[1], p)
    for j=1:size(X)[2]
        X_new[:, 4*(j-1)+1] = X[:, j]
        X_new[:, 4*(j-1)+2] = X[:, j].^2
        X_new[:, 4*(j-1)+3] = sqrt.(abs.(X[:, j]))
        X_new[:, 4*(j-1)+4] = log.(abs.(X[:, j]) .+ e)
    end
    return X_new
end;

function combine_matrices(X1, X2)
    return hcat(X1, X2)
end;

function compute_correlated_pairs(X, rho; output=1)
    corr_pairs = []
    for i=1:size(X)[2]
        for j=i+1:size(X)[2]
            if abs(cor(X[:,i], X[:,j])) > rho
                push!(corr_pairs, (i,j))
            end
        end
    end
    return corr_pairs
end;

function holistic_regression(X, y, lambda, k_vars, rho; solver_output=0)
    n,p = size(X)
    
    # Build model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", solver_output) 
    set_optimizer_attribute(model, "Threads", 48)
    
    # Insert variables
    @variable(model, beta[j=1:p])
    @variable(model, a[j=1:p]>=0)
    @variable(model, t>=0)
    @variable(model, z[j=1:p], Bin)
    
    #Insert constraints
    #Linearize norm 1
    @constraint(model,[j=1:p], beta[j]<=a[j])
    @constraint(model,[j=1:p], -beta[j]<=a[j])

    #The residual term
    @constraint(model, sum((y[i]-sum(X[i,j]*beta[j] for j=1:p))^2 for i=1:n) <= t)

    # Sparsity
    M = 100
    @constraint(model, [j=1:p], -M*z[j] <= beta[j])
    @constraint(model, [j=1:p], beta[j] <= M*(z[j]))

    # # Normal variables
    @constraint(model, sum(z[j] for j=1:p) <= k_vars)

    # # Nonlinear transformations
    @constraint(model, [j=1:p ÷ 4], sum(z[i] for i=4*(j-1)+1:4*j) <= 1)

    # Correlation
    pairs = compute_correlated_pairs(X, rho, output=0)
    for (i, j) in pairs
        @constraint(model, z[i] + z[j] <= 1)
    end
    
    #Objective
    @objective(model, Min, t + lambda*sum(a[j] for j=1:p))
    
    # Optimize
    optimize!(model)
    
    # Return estimated betas
    return (value.(beta), objective_value(model), value.(z))
end;

X_train_new = get_transformation(X_train, 1);
X_valid_new = get_transformation(X_valid, 1);
X_test_new = get_transformation(X_test, 1);

# get the names of the features
feature_names = names(train[:, Not(:SCORE_DIFF)])
# add feature names for transformations
new_feats = []
for i in 1:length(feature_names)
    push!(new_feats, feature_names[i])
    push!(new_feats, string(feature_names[i], "^2"))
    push!(new_feats, string("sqrt(abs(", feature_names[i], "))"))
    push!(new_feats, string("log(abs(", feature_names[i], "))"))
end
feature_names = new_feats;

MSEs = Vector{Float64}(undef, 0)
features = []
results = DataFrame(sparsity = Int64[], lambda = Float64[], MSE = Float64[], features = String[])
for i = 1:6
    for lambda in (0.1, 0.5, 1, 10, 50)
        println("Sparsity: ", i, ", lambda: ", lambda)
        flush(stdout)
        # X, y, lambda, k_vars, rho, num_transformations; solver_output=0
        beta, obj, z_opt = holistic_regression(X_train_new, y_train, lambda, i, 0.7, solver_output=0);
        mse = compute_mse(X_valid_new, y_valid, beta)
        push!(MSEs, mse)
        featrs = findall(x -> x > 0, z_opt);
        push!(features, feature_names[featrs])
        push!(results, (i, mse, lambda, join(feature_names[featrs], ", ")))
        # write beta to file
        open("/home/gbonheur/edge/beta/beta_$i.$lambda.txt", "w") do io
            for i in 1:length(beta)
                write(io, string(feature_names[i], " ", beta[i], "\n"))
            end
        end
        # write to file
        CSV.write("/home/gbonheur/edge/results_cv.csv", results)
    end
end

println(MSEs)
println(features)

# get best sparsity and lambda
best_sparsity = results[findmin(results[:MSE])[2], :sparsity]
best_lambda = results[findmin(results[:MSE])[2], :lambda]
println(best_sparsity)
println(best_lambda)

# print min mse
println(findmin(MSEs))

# get best features
best_features = results[findmin(results[:MSE])[2], :features]
println(best_features)

# run best model on test set
X_train_new = get_transformation(X_train, 1);
X_test_new = get_transformation(X_test, 1);
beta, obj, z_opt = holistic_regression(X_train_new, y_train, best_lambda, best_sparsity, 0.7, solver_output=0);

# compute mse on test set
mse = compute_mse(X_test_new, y_test, beta)
println(mse)