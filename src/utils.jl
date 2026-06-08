using Random, Statistics, NamedArrays

mse(y, y_hat) = norm(y .- y_hat)^2
mse_grad(y, y_hat) = 2 .* (y_hat .- y)

cross_entropy(y, y_hat) = -sum(y .* log.(y_hat))
cross_entropy_grad(y, y_hat) = -1 .* y ./ y_hat

sigmoid(x::Real) = 1 / (1 + exp(-x))
deriv_sigmoid(x::Real) = sigmoid(x) * (1 - sigmoid(x))

# Create 80/20 split
function split(X, y; dims=1, ratio_train = 0.8)
    n = length(y)
    size(X, dims) == n || throw(DimensionMismatch("..."))
    n_train = round(Int, n * ratio_train)
    indices = randperm(n)
    indices_train = indices[1:n_train]
    indices_test = indices[n_train+1:end]
    
    X1 = selectdim(X, dims, indices_train)
    y1 = y[indices_train]    
    X2 = selectdim(X, dims, indices_test)
    y2 = y[indices_test]
    return X1, y1, X2, y2
end

function normalize(X, y; dims = 1)
    X_mean = mean(X, dims = dims)
    y_mean = mean(y, dims = dims)
    X_sd = std(X, dims = dims)
    y_sd = std(y, dims = dims)
    
    X_norm = (X .- X_mean) ./ X_sd
    y_norm = (y .- y_mean) ./ y_sd

    return X_norm, y_norm
end

function onehot(y, classes)
    y_onehot = falses(length(classes), length(y))
    for (i, class) in enumerate(classes)
        y_onehot[i, y .== class] .= 1
    end

    return y_onehot
end

onecold(y, classes) = [classes[argmax(y_col)] for y_col in eachcol(y)]

function prepare_data(X, y; do_normal = true, do_onehot = true, kwargs...)
    X_train, y_train, X_test, y_test = split(X, y; kwargs...)

    if do_normal
        X_train, X_test = normalize(X_train, X_test; kwargs...)
    end
    classes = unique(y)

    if do_onehot
        y_train = Float64.(onehot(y_train, classes))
        y_test = Float64.(onehot(y_test, classes))
    end

    return X_train, y_train, X_test, y_test, classes
end

function confmat(predictions, targets, classes)
    predictions = convert(typeof(classes), predictions)
    targets = convert(typeof(classes), targets)    
    
    length(predictions) == length(targets) || throw(DimensionMismatch("..."))
    all(in.(predictions), classes) || all(in.(targets), classes) || error("Not all classes are included.")

    nclasses = length(classes)
    confmat = NamedArray(zeros(Int64, nclasses, nclasses), (classes, classes), ("True", "Predicted"))

    for i in eachindex(predictions)
        confmat[Name(targets[i]), Name(predictions[i])] += 1
    end
    return confmat
end
