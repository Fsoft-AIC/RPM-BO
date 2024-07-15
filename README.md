# HIGH-DIMENSIONAL BAYESIAN OPTIMIZATION VIA RANDOM PROJECTION OF MANIFOLD SUBSPACES

High-dimensional Bayesian Optimization via Random Projection of Manifold Subspaces (ECML PKDD 2024)

## Installation

```
pip install -r requirements.txt
```

## Geometry-aware synthetic experiment

The geometry-aware synthetic experiments contain in the file [geometry_aware_synthetic_exp.py](https://github.com/Fsoft-AIC/RPM-BO/blob/master/geometry_aware_synthetic_exp.py)

#### Example: Running Ackley Sphere with D=500, d=10

```
python geometry_aware_synthetic_exp.py --test_func Ackley_Sphere_1 --rep 20 --trial_itr 300 --initial_n 10 --high_dim 500 --effective_dim 10 --proj_dim 15 --update_param 3
```

## Geometry-unaware synthetic experiment

The geometry-unaware experiments contain in the file [geometry_unaware_synthetic_exp.py](https://github.com/Fsoft-AIC/RPM-BO/blob/master/geometry_unaware_synthetic_exp.py)

#### Example: Running Ackley Mix with D=500, d=15

```
python geometry_unaware_synthetic_exp.py --test_func Ackley_Mix --rep 20 --trial_itr 300 --initial_n 10 --high_dim 500 --effective_dim 15 --proj_dim 15 --update_param 3
```

## LassoBench experiment

The LassoBench experiments contain in the file [laso_exp.py](https://github.com/Fsoft-AIC/RPM-BO/blob/master/lasso_exp.py)

#### Example: Running Lasso Hard with D=1000

```
python lasso_exp.py --test_func Lasso --rep 20 --trial_itr 300 --initial_n 10 --proj_dim 10 --update_param 3
```

## References

This source code is adopted from:
- [LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso](https://github.com/ksehic/LassoBench)
- [High-dimensional Bayesian Optimization via Nested Riemannian Manifolds](https://github.com/NoemieJaquier/GaBOtorch)
