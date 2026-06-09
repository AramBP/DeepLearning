abstract type NeuralNet end

mutable struct Perceptron <: NeuralNet
    params::Vector{Matrix{Float64}}
    grad_loss::Function
    activation_fun::Function
    deriv_activation_fun::Function
end

append_bias_column(a::Matrix) = cat(-ones(size(a)[1]), a; dims=2) 

function Perceptron(nlayers::Int64, nhidden::Int64, nIn::Int64, nOut::Int64,
    grad_loss::Function, activation_fun::Function, deriv_activation_fun::Function)
    # Initialize the params with small random values
    params = Vector{Matrix{Float64}}(undef, nlayers)
    params[begin] = randn(Float64, nIn+1, nhidden) .* sqrt(2.0 / (nIn + nhidden))
    for i in 2:(nlayers-1)
        params[i] = (randn(Float64, nhidden+1, nhidden).^2) .* sqrt(1/nhidden)
    end
    params[end] = (randn(Float64, nhidden+1, nOut).^2) .* sqrt(2.0 / (nhidden + nOut))

    return Perceptron(params, grad_loss, activation_fun, deriv_activation_fun) 
end

function predict(m::Perceptron, input::Matrix)
    nlayers = length(m.params)
    activations = Vector{Matrix{Float64}}(undef, nlayers + 1)
    pre_activations = Vector{Matrix{Float64}}(undef, nlayers)
    
    activations[1] = input
    for i in 1:nlayers
        pre_activations[i] = append_bias_column(activations[i]) * m.params[i] 
        activations[i+1] = m.activation_fun.(pre_activations[i])
    end

    return activations[end]
end

# gradient through backwards propagation
function grads(m::Perceptron, input::Matrix, target::Matrix)
    nlayers = length(m.params)

    # Forward pass
    activations = Vector{Matrix{Float64}}(undef, nlayers + 1)
    pre_activations = Vector{Matrix{Float64}}(undef, nlayers)
    
    activations[1] = input
    for i in 1:nlayers
        pre_activations[i] = append_bias_column(activations[i]) * m.params[i] 
        activations[i+1] = m.activation_fun.(pre_activations[i])
    end

    # backwards pass
    grads = Vector{Matrix{Float64}}(undef, nlayers)
    delta = m.grad_loss(target, activations[end]) .* m.deriv_activation_fun.(pre_activations[end])

    grads[end] = transpose(append_bias_column(activations[end-1])) * delta

    for i in (nlayers - 1):-1:1
        delta_full = delta * transpose(m.params[i+1])
        delta = delta_full[:, 2:end] .* m.deriv_activation_fun.(pre_activations[i])
        grads[i] = transpose(append_bias_column(activations[i])) * delta
    end

    return grads 
end

mutable struct 