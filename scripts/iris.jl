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
m, xs = train_perceptron(3, X_train, y_train, mse, mse_grad, sigmoid, deriv_sigmoid, gd; max_it = 1000)

predictions = onecold(predict(m, X_test), classes)
targets = onecold(y_test, classes) 
cf = confmat(predictions, targets, classes)

loss_vals = []
for x in xs
    m.params = x
    append!(loss_vals, mse(y_test, predict(m, X_test)))
end
plot(1:length(loss_vals),loss_vals)