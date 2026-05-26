mse(y, y_hat) = norm(y - y_hat)^2
mse_grad(y, y_hat) = 2 .* (y_hat-y)

cross_entropy(y, y_hat) = -sum(y .* log.(y_hat))
cross_entropy_grad(y, y_hat) = -1 .* y ./ y_hat

abstract type Step end 

function optim(f, g, s::Step, x; max_it = 1000, convergence_cond = _ -> false)
    xs = zeros(length(x), max_it+1)
    xs[:,1] = x
    x = x
    for i in 1:max_it
        x = optim_step(s, f, g, x)
        xs[:, i+1] = x
        if (convergence_cond(x))
            break
        end
    end
    return xs
end

struct GD <: Step
    alpha::Float64
end

optim_step(s::GD, f, g, x) = x - s.alpha * g(x)

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
