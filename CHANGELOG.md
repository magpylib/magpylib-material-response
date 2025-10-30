# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2025-10-30

### Fixed

- Fixed susceptibility tree traversal
  ([#26](https://github.com/magpylib/magpylib-material-response/pull/26),
  [#28](https://github.com/magpylib/magpylib-material-response/pull/28))

### Changed

- Improved clarity and error handling in `get_susceptibilities` function
- Enhanced logging configuration
- Refactored demag.py for better organization
- Improved documentation theme and layout
  ([#29](https://github.com/magpylib/magpylib-material-response/pull/29))
  - Switched from Furo to PyData Sphinx theme
  - Added custom CSS for full-width content and smaller sidebar
  - Enhanced navigation and branding with Magpylib logo
  - Reorganized examples section structure

### Maintenance

- Updated pre-commit hooks to latest versions
  ([#21](https://github.com/magpylib/magpylib-material-response/pull/21))
- Updated GitHub Actions dependencies
  ([#24](https://github.com/magpylib/magpylib-material-response/pull/24))
- Removed test PyPI repository URL from publish step
  ([#27](https://github.com/magpylib/magpylib-material-response/pull/27))
- Added VSCode settings for pytest configuration

## [0.3.0] - 2024-10-16

### Added

- Support for anisotropic susceptibilities
- FE tests for anisotropic and negative susceptibility values
- Comprehensive tests for interface and computation validation
- Susceptibility input to parent collections
- Support for both scalar (isotropic) and 3-vector (anisotropic) susceptibility
  inputs

### Changed

- Improved interface for susceptibility (chi) input
- Enhanced susceptibility input possibilities:
  - Scalar input now assumes isotropic susceptibility
  - 3-vector input enables anisotropic susceptibility
- Improved internal architecture and code organization

### Fixed

- Various computation fixes confirmed by isotropic and anisotropic tests

## [0.2.1a0]

### Fixed

- Fixed null polarization for rotated objects
  ([#7](https://github.com/magpylib/magpylib-material-response/pull/7))
- Fixed documentation build issues
  ([#6](https://github.com/magpylib/magpylib-material-response/pull/6))

## [0.2.0a0]

### Changed

- Renamed `xi` parameter to `susceptibility` for better clarity
  ([#5](https://github.com/magpylib/magpylib-material-response/pull/5))
- Updated to support magpylib v5
  ([#4](https://github.com/magpylib/magpylib-material-response/pull/4))

[Unreleased]:
  https://github.com/magpylib/magpylib-material-response/compare/v0.3.1...HEAD
[0.3.1]:
  https://github.com/magpylib/magpylib-material-response/compare/v0.3.0...v0.3.1
[0.3.0]:
  https://github.com/magpylib/magpylib-material-response/compare/v0.2.1a0...v0.3.0
[0.2.1a0]:
  https://github.com/magpylib/magpylib-material-response/compare/v0.2.0a0...v0.2.1a0
[0.2.0a0]:
  https://github.com/magpylib/magpylib-material-response/releases/tag/v0.2.0a0
