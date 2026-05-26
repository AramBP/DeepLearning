module Perceptron
export init_perceptron, grads
using LinearAlgebra


activation_func(x::Real) = 1 / (1 + exp(-x))
deriv_activation_func(x::Real) = activation_func(x) * (1 - activation_func(x))
append_bias_column(a::Matrix) = cat(-ones(size(a)[1]), a; dims=2) 

function init_weights(nlayers::Int64, nhidden::Int64, nIn::Int64, nOut::Int64)
    # Initialize the weights with small random values
    weights = Vector{Matrix{Float64}}(undef, nlayers)
    weights[begin] = randn(Float64, nIn+1, nhidden) .* sqrt(2.0 / (nIn + nhidden))
    for i in 2:(nlayers-1)
        weights[i] = (randn(Float64, nhidden+1, nhidden).^2) .* sqrt(1/nhidden)
    end
    weights[end] = (randn(Float64, nhidden+1, nOut).^2) .* sqrt(2.0 / (nhidden + nOut))

    return weights
end

function predict(weights::Vector{Matrix{Float64}}, input::Matrix)
    activations = Vector{Matrix{Float64}}(undef, nlayers + 1)
    pre_activations = Vector{Matrix{Float64}}(undef, nlayers)
    
    activations[1] = input
    for i in 1:nlayers
        pre_activations[i] = append_bias_column(activations[i]) * weights[i] 
        activations[i+1] = activation_func.(pre_activations[i])
    end
    return activations[end]
end

# gradient through backwards propagation
function grads(weights::Vector{Matrix{Float64}}, input::Matrix, target::Matrix, grad_loss)
    nlayers = length(weights)

    # Forward pass
    activations = Vector{Matrix{Float64}}(undef, nlayers + 1)
    pre_activations = Vector{Matrix{Float64}}(undef, nlayers)
    
    activations[1] = input
    for i in 1:nlayers
        pre_activations[i] = append_bias_column(activations[i]) * weights[i] 
        activations[i+1] = activation_func.(pre_activations[i])
    end

    # backwards pass
    grads = Vector{Matrix{Float64}}(undef, nlayers)
    delta = grad_loss(target, activations[end]) .* deriv_activation_func.(pre_activations[end])

    grads[end] = transpose(append_bias_column(activations[end-1])) * delta

    for i in (nlayers - 1):-1:1
        delta_full = delta * transpose(weights[i+1])
        delta = delta_full[:, 2:end] .* deriv_activation_func.(pre_activations[i])
        grads[i] = transpose(append_bias_column(activations[i])) * delta
    end

    return grads 
end
end