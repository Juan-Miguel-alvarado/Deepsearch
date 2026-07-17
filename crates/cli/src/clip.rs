//! Copy text to the system clipboard by shelling out to whatever clipboard
//! tool is installed. Avoids pulling in a clipboard crate (and an X/Wayland
//! build dependency) for a one-shot copy.
//!
//! Detection order matches the common setups: Wayland (`wl-copy`), X11
//! (`xclip`/`xsel`), macOS (`pbcopy`), and Windows (`clip`).

use std::io::Write;
use std::process::{Command, Stdio};

/// A clipboard tool: the program plus any fixed arguments.
struct Tool {
    program: &'static str,
    args: &'static [&'static str],
}

const TOOLS: &[Tool] = &[
    Tool {
        program: "wl-copy",
        args: &[],
    },
    Tool {
        program: "xclip",
        args: &["-selection", "clipboard"],
    },
    Tool {
        program: "xsel",
        args: &["--clipboard", "--input"],
    },
    Tool {
        program: "pbcopy",
        args: &[],
    },
    Tool {
        program: "clip",
        args: &[],
    },
];

/// Copy `text` to the clipboard. Returns the tool used on success, or an error
/// message describing why it couldn't.
pub fn copy(text: &str) -> Result<&'static str, String> {
    let Some(tool) = TOOLS.iter().find(|t| on_path(t.program)) else {
        return Err("no clipboard tool found (install wl-clipboard, xclip or xsel)".to_string());
    };

    let mut child = Command::new(tool.program)
        .args(tool.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to run {}: {e}", tool.program))?;

    if let Some(stdin) = child.stdin.take() {
        // Drop the handle after writing so the tool sees EOF and exits.
        let mut stdin = stdin;
        stdin
            .write_all(text.as_bytes())
            .map_err(|e| format!("failed to write to {}: {e}", tool.program))?;
    }

    match child.wait() {
        Ok(status) if status.success() => Ok(tool.program),
        Ok(status) => Err(format!("{} exited with {status}", tool.program)),
        Err(e) => Err(format!("{} failed: {e}", tool.program)),
    }
}

fn on_path(bin: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|dir| dir.join(bin).is_file())
}
