using Pkg
cd(joinpath(@__DIR__, ".."))
Pkg.activate(".")

using RDatasets
using DeepLearning
using Plots

iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])
y = convert(Vector{String}, iris.Species)

X_train, y_train, X_test, y_test, classes = prepare_data(X, y)
gd = GradientDescent(0.001)
m_gd, xs_gd = train_perceptron(3, X_train, y_train, mse, mse_grad, sigmoid, deriv_sigmoid, gd; max_it = 1000)
predictions_gd = onecold(predict(m_gd, X_test), classes)

sgd = StochasticGradientDescent(0.001, 16)
m_sgd, xs_sgd = train_perceptron(3, X_train, y_train, mse, mse_grad, sigmoid, deriv_sigmoid, sgd; max_it = 100)
predictions_sgd = onecold(predict(m_sgd, X_test), classes)

targets = onecold(y_test, classes) 

loss_vals_gd = []
for x in xs_gd
    m_gd.params = x
    append!(loss_vals_gd, mse(y_test, predict(m_gd, X_test)))
end

loss_vals_sgd = []
for x in xs_sgd
    m_sgd.params = x
    append!(loss_vals_sgd, mse(y_test, predict(m_sgd, X_test)))
end

plot(1:length(loss_vals_gd),loss_vals_gd)
plot(1:length(loss_vals_sgd), loss_vals_sgd)