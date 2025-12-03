use anyhow::{Context, anyhow};
use log::{self, LevelFilter, Log, Metadata, Record};
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::str::FromStr;
use std::sync::Once;

static LOGGER_INIT_ONCE: Once = Once::new();

struct FileLogger {
    //文件加锁会死锁，simple_logger也会死锁，所以这里不加锁
    file: std::fs::File,
    level: LevelFilter,
}

impl Log for FileLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let msg = format!("[{}] {}\n", record.target(), record.args());
        let _ = (&self.file).write_all(msg.as_bytes());
    }

    fn flush(&self) {
        let _ = (&self.file).flush();
    }
}

fn init_file_logger(path: &str, level: LevelFilter) -> Result<(), anyhow::Error> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open log file {}", path))?;

    let logger = FileLogger {
        file,
        level,
    };

    log::set_boxed_logger(Box::new(logger))
        .map_err(|e| anyhow!("failed to set global logger: {}", e))?;
    log::set_max_level(level);

    Ok(())
}

pub fn init_logger() {
    LOGGER_INIT_ONCE.call_once(|| {
        match env::var("HANGDETECT_LOG_FILE") {
            Ok(log_file) => {
                if let Err(err) = makedirs_for_file(&log_file) {
                    env_logger::init();
                    return;
                }
                let local_rank = env::var("LOCAL_RANK").unwrap_or_else(|_| "0".to_string());
                let log_file = format!("{}.{}", log_file, local_rank);

                let level = env::var("HANGDETECT_LOG_LEVEL").unwrap_or_else(|_| "info".to_string());

                let level = match LevelFilter::from_str(&level) {
                    Ok(level) => level,
                    Err(e) => {
                        LevelFilter::Info
                    }
                };

                if let Err(err) = init_file_logger(&log_file, level) {
                    env_logger::init();
                }
            }
            Err(_) => {
                env_logger::init();
            }
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
