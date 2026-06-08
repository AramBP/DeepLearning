abstract type Optimizer end

struct GradientDescent <: Optimizer
    alpha::Float64
end

function optim(gd::GradientDescent, m::NeuralNet, loss, grad; max_it = 1000)
    theta = Vector{typeof(m.params)}(undef, max_it+1)
    theta[1] = m.params
    for i in 2:max_it
        m.params = m.params .- gd.alpha .* grad(m) 
        theta[i] = m.params
    end
    return (m, theta)
end

function train_perceptron(nlayers, input, target, loss, grad_loss, activation_fun, deriv_activation_fun, optimizer::Optimizer; max_it = 1000)
    nIn = size(input, 2)
    nOut = size(target, 2) 

    nhidden = round(Int64, 0.5*(nIn + nOut))
    m = Perceptron(nlayers, nhidden, nIn, nOut, grad_loss, activation_fun, deriv_activation_fun)
    grad_perceptron(m) = grads(m, input, target)
    (m, theta) = optim(optimizer, m, loss, grad_perceptron; max_it = max_it)
    return (m, theta)
end
