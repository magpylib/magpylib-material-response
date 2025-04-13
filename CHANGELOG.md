# Unreleased

# 0.3.0

- improved interface for chi input
- FE tests for anisotropic chi and negative chi
- Improve internals
- anisotropic susceptibilities are now allowed.
- Improve suszeptibility input possibilities:
  - give susceptibility to parent collection
  - if susceptiblity input is scalar, isotropic susceptibility is assumed, if it
    is a 3-vector it can be anisotropic
- Various tests included of interface and computation, isotropic and anisotropic
  tests confirm computaiton

# 0.2.1a0

- Fix null polarization for rotated objects
  ([#7](https://github.com/magpylib/magpylib-material-response/pull/7))
- Fix docs not building
  ([#6](https://github.com/magpylib/magpylib-material-response/pull/6))

# 0.2.0a0

- renaming xi->susceptibility
  ([#5](https://github.com/magpylib/magpylib-material-response/pull/5))
- update to magpylib v5
  ([#4](https://github.com/magpylib/magpylib-material-response/pull/4))
