using Random, Statistics

# Create 80/20 split
function split(X, y; dims=1, ratio_train = 0.8)
    n = size(X)[1]
    size(X, dims) == n || throw(DimensionMismatch("..."))
    n_train = round(Int, n * ratio_train)
    indices = randperm(n)
    indices_train = indices[1:n_train]
    indices_test = indices[n_train+1:end]
    
    X1 = selectdim(X, dims = dims, indices_train)
    y1 = selectdim(y, dims = dims, indices_train)
    X2 = selectdim(X, dims = dims, indices_test)
    y2 = selectdim(y, dims = dims, indices_train)
    return X1, y1, X2, y2
end

function normalize(X, y; dims = 1)
    X_mean = mean(X, dims = dims)
    y_mean = mean(y, dims = dims)
    X_sd = sd(X, dims = dims)
    y_sd = sd(y, dims = dims)
    
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
    X_train, y_train, X_test, y_test = split(X, y; kwargs)

    if do_normal
        x_train, X_test = normalize(X_train, X_test; kwargs)
    end
    classes = unique(y)

    if do_onehot
        y_train = onehot(y_train, classes)
        y_test = onehot(y_test, classes)
    end

    return X_train, y_train, X_test, y_test
end

function confmat(outputs, targets)

end

