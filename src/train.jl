include("neural_net.jl")

abstract type Step end 

function train_perceptron(nlayers, input, target, loss, grad_loss, s::Step; max_it = 1000, convergence_cond = _ -> false)
    nIn = size(input, 2)
    nOut = size(target, 2) 
    @show (nIn, nOut)
    nhidden = round(Int64, 0.5*(nIn + nOut))
    theta0 = Perceptron.init_weights(nlayers, nhidden, nIn, nOut)
    
    perceptron_grad(theta) = Perceptron.grads(theta, input, target, grad_loss) 
    theta = optim(loss, perceptron_grad, s, theta0; max_it = max_it, convergence_cond = convergence_cond)
    preds = [predict(x, input) for x in theta]
    return preds
end

function optim(f, g, s::Step, x; max_it = 1000, convergence_cond = _ -> false)
    xs = Vector{typeof(x)}(undef, max_it+1)
    xs[1] = x
    x = x
    for i in 1:max_it
        x = optim_step(s, f, g, x)
        xs[i+1] = x
        if (convergence_cond(x))
            break
        end
    end
    return xs
end

struct GD <: Step
    alpha::Float64
end

optim_step(s::GD, f, g, x) = x .- s.alpha .* g(x)

struct Armijo <: Step
    c::Float64
    alpha_max::Float64
end

function optim_step(s::Armijo, f, g, x) 
    alpha = s.alpha_max
    fun = f(x)
    g = grad(x)
    while f(x - alpha*g) > fun - c * alpha * abs(g'*g) 
        alpha = alpha/2
        if alpha < 1e-6
            warning("Armijo line search failed")
            break
        end
    end
    return x-alpha*g
end
