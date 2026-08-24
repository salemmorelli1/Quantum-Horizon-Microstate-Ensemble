# Security policy

## Supported version

Version `0.1.x` receives security and reproducibility fixes.

## Reporting

Report potential vulnerabilities privately through GitHub's security advisory
feature. Do not include credentials, private datasets, or untrusted serialized
PyTorch files in an issue.

## Serialization

The repository loads generated tensor artifacts with `weights_only=True`.
PyTorch serialization must never be used to load files from an untrusted source.

## Scientific integrity

Silent changes to the probability measure, Jacobian, adaptation schedule,
degeneracy interpretation, or retained-sample definition are treated as
integrity vulnerabilities and must be disclosed explicitly.
