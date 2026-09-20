//! Read PSTR directly; never substitute or combine IOReport component estimates.

use std::error::Error;
use std::io::{self, Write};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use macmon::sources::SMC;

const VERSION: &str = "laya-pstr-sampler 0.1.0 (PSTR-only; macmon 0.8.2)";
const HELP: &str = "Usage: pstr-sampler pipe [-i|--interval MILLISECONDS] [-s|--samples COUNT]\n\
    Read-only SMC PSTR watts as JSON lines. Default: 500 ms, unlimited samples.\n\
    No CPU/GPU/ANE component counters are sampled or emitted.\n\
    --version prints the sampler and pinned source-library versions.";

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Help,
    Version,
    Pipe { interval_ms: u64, samples: u64 },
}

fn parse_command(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut args = args.into_iter();
    match args.next().as_deref() {
        Some("--help" | "-h") | None => return Ok(Command::Help),
        Some("--version" | "-V") => return Ok(Command::Version),
        Some("pipe") => {}
        Some(value) => return Err(format!("Unknown command: {value}")),
    }
    let mut interval_ms = 500;
    let mut samples = 0;
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(Command::Help);
        }
        match flag.as_str() {
            "-i" | "--interval" | "-s" | "--samples" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("Missing value for {flag}"))?
                    .parse::<u64>()
                    .map_err(|_| format!("{flag} requires a nonnegative integer"))?;
                if flag == "-i" || flag == "--interval" {
                    if value == 0 {
                        return Err("Sampling interval must be positive".into());
                    }
                    interval_ms = value;
                } else {
                    samples = value;
                }
            }
            _ => return Err(format!("Unknown option: {flag}")),
        }
    }
    Ok(Command::Pipe {
        interval_ms,
        samples,
    })
}

fn validate_watts(value: f32) -> Result<f32, String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("Invalid original SMC PSTR reading: {value} W"));
    }
    // Preserve every finite positive value. The energy harness separately
    // rejects physically implausible values using its recorded sanity ceiling.
    Ok(value)
}

fn run() -> Result<(), Box<dyn Error>> {
    let command = parse_command(std::env::args().skip(1))?;
    let (interval_ms, samples) = match command {
        Command::Help => {
            println!("{HELP}");
            return Ok(());
        }
        Command::Version => {
            println!("{VERSION}");
            return Ok(());
        }
        Command::Pipe {
            interval_ms,
            samples,
        } => (interval_ms, samples),
    };
    let mut smc = SMC::new()?;
    let clock = Instant::now();
    let mut count = 0_u64;
    let stdout = io::stdout();
    let mut output = stdout.lock();
    loop {
        // read_float_val verifies the key has the 4-byte "flt " SMC type.
        // Errors propagate; missing data never becomes zero or a component sum.
        let watts = validate_watts(smc.read_float_val("PSTR")?)?;
        let sampled_at = clock.elapsed().as_secs_f64();
        writeln!(
            output,
            "{{\"sys_power\":{watts},\"power_source\":\"SMC:PSTR\",\"source_library\":\"macmon=0.8.2\",\"sampler\":\"laya-pstr-sampler=0.1.0\",\"sample_elapsed_seconds\":{sampled_at}}}"
        )?;
        output.flush()?;
        count += 1;
        if samples != 0 && count >= samples {
            break;
        }
        std::thread::sleep(Duration::from_millis(interval_ms));
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("pstr-sampler: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_validation_preserves_values_and_rejects_missing_or_nonfinite_data() {
        for value in [0.01_f32, 24.15, 40089.36] {
            assert_eq!(validate_watts(value).unwrap().to_bits(), value.to_bits());
        }
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(validate_watts(value).is_err());
        }
    }

    #[test]
    fn pipe_options_keep_the_requested_interval_and_sample_count() {
        let parse = |values: &[&str]| parse_command(values.iter().map(|s| s.to_string()));
        assert_eq!(
            parse(&["pipe", "-i", "500", "-s", "3"]),
            Ok(Command::Pipe {
                interval_ms: 500,
                samples: 3
            })
        );
        assert!(parse(&["pipe", "-i", "0"]).is_err());
        assert!(parse(&["pipe", "-i", "oops"]).is_err());
        assert!(parse(&["pipe", "-i"]).is_err());
        assert!(parse(&["pipe", "--unknown"]).is_err());
        assert_eq!(parse(&["--version"]), Ok(Command::Version));
    }
}
