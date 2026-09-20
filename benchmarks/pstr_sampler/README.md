# Direct SMC PSTR sampler

A small read-only JSON sampler for the local energy benchmark. It opens one SMC
connection and repeatedly reads `PSTR` through the public low-level Rust API in
**macmon 0.8.2**, pinned with default features disabled. It does not create an
IOReport sampler, read component power counters, or apply a component-sum floor.

Upstream macmon 0.8.2 computes `sys_power = max(PSTR, all_power)`. One local run
observed an implausible component spike that therefore polluted `sys_power`.
Reading PSTR directly removes that propagation path. It does not turn this private
SMC estimate into an externally calibrated wall-power or battery measurement.

Build and perform a finite read-only smoke check from the repository root:

```sh
cargo build --release --locked --manifest-path benchmarks/pstr_sampler/Cargo.toml
benchmarks/pstr_sampler/target/release/pstr-sampler --version
benchmarks/pstr_sampler/target/release/pstr-sampler pipe -i 500 -s 3
```

Each JSON line contains `sys_power` in watts and source/version metadata.
CPU/GPU/ANE/RAM fields are deliberately absent because this tool does not measure
them. `sample_elapsed_seconds` is the sampler's monotonic time just after reading
the sensor; the benchmark harness retains its own monotonic receive timestamps.
The stream flushes every sample. `-s 0` means unlimited samples.

SMC read errors, non-finite values and nonpositive values exit unsuccessfully.
Finite positive readings are preserved without clipping or substitution. The
energy harness must also apply its recorded plausibility ceiling and invalidate
an entire run if any sample fails; it must not delete inconvenient samples.
All builds, generated executables and unit-test outputs remain in ignored `target/`.

Hardware-free unit checks:

```sh
cargo test --locked --manifest-path benchmarks/pstr_sampler/Cargo.toml
cargo fmt --check --manifest-path benchmarks/pstr_sampler/Cargo.toml
```

The wrapper follows this repository's Apache-2.0 license. It depends on the
MIT-licensed [macmon project](https://github.com/vladkens/macmon/tree/v0.8.2);
the upstream license is reproduced in `LICENSE.macmon`. Relevant upstream source:

- [PSTR and component-sum handling](https://github.com/vladkens/macmon/blob/v0.8.2/src_lib/metrics.rs)
- [SMC read API and type checks](https://github.com/vladkens/macmon/blob/v0.8.2/src_lib/sources.rs)
- [Public sources module](https://github.com/vladkens/macmon/blob/v0.8.2/src_lib/lib.rs)
