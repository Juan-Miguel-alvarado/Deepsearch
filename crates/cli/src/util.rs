//! Small formatting helpers (human sizes, epoch -> date) with no extra deps.

/// Format a byte count like `1.2 MB`.
pub fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}

/// Format a Unix timestamp (seconds, UTC) as `YYYY-MM-DD HH:MM:SS`.
///
/// Uses Howard Hinnant's civil-from-days algorithm so we avoid pulling in a
/// date library for a one-off display string.
pub fn format_timestamp(secs: i64) -> String {
    if secs <= 0 {
        return "unknown".to_string();
    }
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let (y, mon, d) = civil_from_days(days);
    format!("{y:04}-{mon:02}-{d:02} {h:02}:{m:02}:{s:02} UTC")
}

/// Convert days since 1970-01-01 to (year, month, day).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(1024), "1.0 KB");
        assert_eq!(human_size(1_572_864), "1.5 MB");
    }

    #[test]
    fn epoch_zero_is_unknown() {
        assert_eq!(format_timestamp(0), "unknown");
    }

    #[test]
    fn known_timestamp() {
        // 2021-01-01 00:00:00 UTC = 1609459200
        assert_eq!(format_timestamp(1_609_459_200), "2021-01-01 00:00:00 UTC");
    }
}
