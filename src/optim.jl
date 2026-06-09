abstract type Optimizer end

struct GradientDescent <: Optimizer
    alpha::Float64
end

function optim(gd::GradientDescent, m::NeuralNet, loss, grad, input, target; max_it = 1000)
    theta = Vector{typeof(m.params)}(undef, max_it+1)
    theta[1] = m.params
    for i in 2:(max_it+1)
        m.params = m.params .- gd.alpha .* grad(m, input, target) 
        theta[i] = m.params
    end
    return (m, theta)
end

struct StochasticGradientDescent <: Optimizer
    alpha::Float64
    batch_size::Int64
end

function optim(sgd::StochasticGradientDescent, m::NeuralNet, loss, grad, input, target; max_it=1000)
    n_obs = size(input, 1)

    theta = Vector{typeof(m.params)}(undef, max_it+1)
    theta[1] = m.params
    for i in 2:(max_it+1)
        input, target = split(input, target; ratio_train = 1.0)
        for j in 1:sgd.batch_size:n_obs
            n = j+sgd.batch_size-1 > n_obs ? n_obs : j+sgd.batch_size-1
            m.params = m.params .- sgd.alpha .* grad(m, input[j:n,:], target[j:n,:])
        end
        theta[i] = m.params
    end
    return (m, theta)
end

function train_perceptron(nlayers, input, target, loss, grad_loss, activation_fun, deriv_activation_fun, optimizer::Optimizer; max_it = 1000)
    nIn = size(input, 2)
    nOut = size(target, 2) 

    nhidden = round(Int64, 0.5*(nIn + nOut))
    m = Perceptron(nlayers, nhidden, nIn, nOut, grad_loss, activation_fun, deriv_activation_fun)
    (m, theta) = optim(optimizer, m, loss, grads, input, target; max_it = max_it)
    return (m, theta)
end
