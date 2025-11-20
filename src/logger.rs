use anyhow::Context;
use log::LevelFilter;
use std::env;
use std::str::FromStr;
use std::sync::Once;

static LOGGER_INIT_ONCE: Once = Once::new();

pub fn init_logger() {
    LOGGER_INIT_ONCE.call_once(|| match option_env!("HANGDETECT_LOG_FILE") {
        Some(log_file) => {
            if let Err(err) = makedirs_for_file(log_file) {
                eprintln!(
                    "Failed to create directories for log file {}, fall back to env logger: {}",
                    log_file, err
                );
                env_logger::init();
                return;
            }
            let local_rank = env::var("LOCAL_RANK").unwrap_or_else(|_| "0".to_string());
            let log_file = format!("{}.{}", log_file, local_rank);

            let level = option_env!("HANGDETECT_LOG_LEVEL").unwrap_or_else(|| "info");

            let level = match LevelFilter::from_str(level) {
                Ok(level) => level,
                Err(e) => {
                    eprintln!("Invalid log level {}, fall back to info: {}", level, e);
                    LevelFilter::Info
                }
            };

            if let Err(err) = simple_logging::log_to_file(log_file, level) {
                eprintln!(
                    "Failed to init logger to file, fall back to env logger: {}",
                    err
                );
                env_logger::init();
            }
        }
        None => {
            eprintln!("HANGDETECT_LOG_FILE env variable not set, fall back to env logger");
            env_logger::init();
        }
    })
}

fn makedirs_for_file(p0: &str) -> Result<(), anyhow::Error> {
    use std::fs;
    use std::path::Path;
    let path = Path::new(p0);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directories for log file {}", p0))?;
    }
    Ok(())
}
