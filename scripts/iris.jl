using Pkg
cd(joinpath(@__DIR__, ".."))
Pkg.activate(".")

using RDatasets
using DeepLearning
using Plots

t(X) = copy(transpose(X))

iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])
y = convert(Vector{String}, iris.Species)

X_train, y_train, X_test, y_test, classes = prepare_data(transpose(X), y; dims = 2)
gd = GradientDescent(0.1)
m, xs = train_perceptron(3, t(X_train), t(y_train), mse, mse_grad, sigmoid, deriv_sigmoid, gd; max_it = 200)
cf = confmat(onecold(t(predict(m, t(X_test))), classes), onecold(y_test, classes), classes)

loss_vals = []
for x in xs
    m.params = x
    append!(loss_vals, mse(t(y_test), predict(m, t(X_test))))
end

plot(1:length(loss_vals),loss_vals)