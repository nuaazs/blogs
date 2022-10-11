The following are a set of methods intended for regression in which the target value is expected to be a linear combination of the features. In mathematical notation, if $$ \hat{y} $$ is the predicted value.
$$
 \hat{y}(w, x)=w_{0}+w_{1} x_{1}+\ldots+w_{p} x_{p} 
$$
Across the module, we designate the vector $$ w=\left(w_{1}, \ldots, w_{p}\right) $$ as `coef_` and $$ w_{0} $$ as `intercept_`.

## 最小二乘法 Ordinary Least Squares

$$ \min _{w}\|X w-y\|_{2}^{2} $$

