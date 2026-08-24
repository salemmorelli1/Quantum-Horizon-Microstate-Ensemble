# Generated artifacts

The validated Part 0→Part 4 pipeline writes machine-readable outputs here.
Generated subdirectories are excluded from version control because every result
must be reproducible from a committed configuration, package version, and seed.

Expected outputs:

```text
part0/model_validation.json
part1/samples.pt
part1/hmc_summary.json
part2/quantization_summary.json
part3/geometry_summary.json
part4/validation_summary.json
```

Tensor artifacts are loaded with `weights_only=True`. Never load untrusted
PyTorch serialization files.
