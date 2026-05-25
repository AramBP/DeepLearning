module Perceptron
export init_perceptron, grads
using LinearAlgebra

struct DeepPerceptron
    loss::Function
    grad_loss::Function
    weights::Vector{Matrix{Float64}}
end

function init_perceptron(nlayers::Int64, nhidden::Int64, X::Matrix{Float64}, y::Vector{Float64}, loss::Function, grad_loss::Function)
    weights = Vector{Matrix{Float64}}(undef, nlayers)

    return DeepPerceptron(loss, grad_loss, weights)
end

activation_func(x::Real) = 1 / (1 + exp(-x))
deriv_activation_func (x::Real) = activation_func(x) * (1 - activation_func(x))
append_bias_column(a::Matrix) = cat(-ones(size(a)[1]), a; dims=2) 

# gradient through backwards propagation
function grads(m::Perceptron, input::Matrix, target::Vector)
    nlayers = length(m.weights)

    # Forward pass
    activations = Vector{Matrix{Float64}}(undef, nlayers + 1)
    pre_activations = Vector{Matrix{Float64}}(undef, nlayers)
    
    activations[1] = input
    for i in 1:nlayers
        pre_activations[i] = append_bias_column(activations[i]) * transpose(m.weights[i]) 
        activations[i+1] = activation_func.(pre_activations[i])
    end

    # backwards pass
    grads = Vector{Matrix{Float64}}(undef, nlayers)
    delta = m.grad_loss.(activations[end]) .* deriv_activation_func.(pre_activations[end])

    grads[end] = transpose(append_bias_column(activations[end-1])) * delta

    for i in (nlayers - 1):-1:1
        delta_full = delta * transpose(m.weights[i])
        delta = delta_full[:, 2:end] .* deriv_activation_func.(pre_activations[i])
        grads[i] = transpose(append_bias_column(activations[i])) * delta
    end

    return grads 
end
end