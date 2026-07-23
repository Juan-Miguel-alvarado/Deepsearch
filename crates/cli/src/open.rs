//! "Open with" — detect installed applications and build the list of ways the
//! user can open the selected file.
//!
//! Detection is by presence on `PATH`, so the menu only ever offers apps that
//! are actually installed. Each candidate carries the exact command to run
//! (program + args, with the target path already baked in) and an [`AppKind`]
//! that drives its menu colour and the "smart open" default. Terminal apps
//! (editors like vim) take over the screen — the TUI suspends itself around
//! them; GUI apps are spawned detached.

use std::path::Path;

use deepsearch_core::FileType;

/// What kind of app a menu entry is — drives its menu colour and whether "smart
/// open" (Enter) treats it as a fallback editor or a real viewer.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AppKind {
    Editor,
    Image,
    Pdf,
    Media,
    Default,
    Reveal,
    Terminal,
}

/// One launchable way to open a file: a ready-to-run command.
#[derive(Clone)]
pub struct OpenApp {
    /// Human-readable name shown in the menu.
    pub label: String,
    pub kind: AppKind,
    /// Executable to run.
    pub program: String,
    /// Full argument list, including the target path/dir.
    pub args: Vec<String>,
    /// Terminal apps take over the screen (the caller must suspend the TUI);
    /// GUI apps are spawned detached.
    pub terminal: bool,
}

/// Internal category used to pick and order the known viewers/editors.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cat {
    Editor,
    Image,
    Pdf,
    Media,
}

impl Cat {
    fn kind(self) -> AppKind {
        match self {
            Cat::Editor => AppKind::Editor,
            Cat::Image => AppKind::Image,
            Cat::Pdf => AppKind::Pdf,
            Cat::Media => AppKind::Media,
        }
    }
}

/// Known applications, checked against `PATH`. Order within a category is the
/// order they appear in the menu.
const KNOWN: &[(&str, &str, bool, Cat)] = &[
    // Editors / IDEs.
    ("code", "VS Code", false, Cat::Editor),
    ("code-insiders", "VS Code Insiders", false, Cat::Editor),
    ("cursor", "Cursor", false, Cat::Editor),
    ("zed", "Zed", false, Cat::Editor),
    ("zeditor", "Zed", false, Cat::Editor),
    ("subl", "Sublime Text", false, Cat::Editor),
    ("nvim", "Neovim", true, Cat::Editor),
    ("vim", "Vim", true, Cat::Editor),
    ("hx", "Helix", true, Cat::Editor),
    ("nano", "nano", true, Cat::Editor),
    // Image viewers.
    ("imv", "imv", false, Cat::Image),
    ("feh", "feh", false, Cat::Image),
    ("eog", "Image Viewer", false, Cat::Image),
    ("gwenview", "Gwenview", false, Cat::Image),
    ("gimp", "GIMP", false, Cat::Image),
    // PDF viewers.
    ("zathura", "Zathura", false, Cat::Pdf),
    ("evince", "Evince", false, Cat::Pdf),
    ("okular", "Okular", false, Cat::Pdf),
    ("xpdf", "xpdf", false, Cat::Pdf),
    // Media players (video/audio).
    ("mpv", "mpv", false, Cat::Media),
    ("vlc", "VLC", false, Cat::Media),
];

const VIDEO_EXTS: &[&str] = &[
    "mp4", "mkv", "webm", "mov", "avi", "flv", "wmv", "m4v", "mpg", "mpeg",
];
const AUDIO_EXTS: &[&str] = &["mp3", "flac", "wav", "ogg", "m4a", "aac", "opus", "wma"];

/// Build the ordered list of apps offered for `path` (of type `file_type`).
///
/// Order: viewers for the file's own kind first, then editors, the OS default
/// handler, `$EDITOR`, and finally the location actions (reveal / terminal).
pub fn candidates_for(path: &Path, file_type: FileType) -> Vec<OpenApp> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let target = path.to_string_lossy().to_string();

    let primary = primary_category(file_type, &ext);

    let mut apps: Vec<OpenApp> = Vec::new();
    let mut seen: Vec<String> = Vec::new();

    // 1. Apps for the file's own category first (image viewer for an image, ...).
    if let Some(cat) = primary {
        push_category(&mut apps, &mut seen, cat, &target);
    }
    // 2. Editors are useful for almost anything; offer them next.
    if primary != Some(Cat::Editor) {
        push_category(&mut apps, &mut seen, Cat::Editor, &target);
    }
    // 3. The OS default handler — respects the user's mimetype associations.
    if let Some(mut def) = default_opener() {
        def.args.push(target.clone());
        push_unique(&mut apps, &mut seen, def);
    }
    // 4. `$EDITOR` as a guaranteed terminal fallback.
    if let Ok(ed) = std::env::var("EDITOR") {
        let ed = ed.trim().to_string();
        if !ed.is_empty() {
            push_unique(
                &mut apps,
                &mut seen,
                OpenApp {
                    label: format!("$EDITOR ({ed})"),
                    kind: AppKind::Editor,
                    program: ed,
                    args: vec![target.clone()],
                    terminal: true,
                },
            );
        }
    }
    // 5. Location actions. For a directory the location *is* the item, so
    //    "terminal here" opens inside it rather than in its parent.
    let location = if file_type == FileType::Dir {
        Some(path.to_path_buf())
    } else {
        path.parent().map(|p| p.to_path_buf())
    };
    if let Some(parent) = location {
        let dir = parent.to_string_lossy().to_string();
        if let Some(mut rev) = default_opener() {
            rev.label = if file_type == FileType::Dir {
                "Open folder".to_string()
            } else {
                "Reveal in folder".to_string()
            };
            rev.kind = AppKind::Reveal;
            rev.args.push(dir.clone());
            apps.push(rev);
        }
        if let Some(term) = terminal_here(&dir) {
            apps.push(term);
        }
    }

    apps
}

/// The category whose viewers should be listed first for this file, or `None`
/// when nothing specific fits (Word docs, generic binaries) — those are best
/// served by the OS default handler.
fn primary_category(ft: FileType, ext: &str) -> Option<Cat> {
    match ft {
        FileType::Image => Some(Cat::Image),
        FileType::Pdf => Some(Cat::Pdf),
        FileType::Text | FileType::Code => Some(Cat::Editor),
        FileType::Docx => None,
        // A folder is best handed to the file manager (the OS default), though
        // editors can open it as a project, so they stay in the list.
        FileType::Dir => None,
        // Type is content-based and doesn't distinguish video/audio from other
        // binaries, so fall back to the extension purely for menu ordering.
        FileType::Binary => {
            if VIDEO_EXTS.contains(&ext) || AUDIO_EXTS.contains(&ext) {
                Some(Cat::Media)
            } else {
                None
            }
        }
    }
}

/// Append every installed app in `cat` (skipping ones already listed).
fn push_category(apps: &mut Vec<OpenApp>, seen: &mut Vec<String>, cat: Cat, target: &str) {
    for &(bin, label, terminal, kcat) in KNOWN {
        if kcat == cat && on_path(bin) {
            push_unique(
                apps,
                seen,
                OpenApp {
                    label: label.to_string(),
                    kind: cat.kind(),
                    program: bin.to_string(),
                    args: vec![target.to_string()],
                    terminal,
                },
            );
        }
    }
}

fn push_unique(apps: &mut Vec<OpenApp>, seen: &mut Vec<String>, app: OpenApp) {
    if seen.iter().any(|p| p == &app.program) {
        return;
    }
    seen.push(app.program.clone());
    apps.push(app);
}

/// The platform's default "open this in whatever is associated" launcher, with
/// no target appended yet (the caller pushes the path or dir).
fn default_opener() -> Option<OpenApp> {
    let (program, args): (&str, Vec<String>) = if cfg!(target_os = "macos") {
        ("open", Vec::new())
    } else if cfg!(target_os = "windows") {
        // `cmd /C start "" <target>` uses the shell's file association.
        (
            "cmd",
            vec!["/C".to_string(), "start".to_string(), String::new()],
        )
    } else if on_path("xdg-open") {
        ("xdg-open", Vec::new())
    } else {
        return None;
    };
    Some(OpenApp {
        label: "Default app".to_string(),
        kind: AppKind::Default,
        program: program.to_string(),
        args,
        terminal: false,
    })
}

/// A "terminal here" entry that opens the user's terminal in `dir`, if a known
/// terminal is installed. Each terminal spells its working-directory flag
/// differently, hence the table.
fn terminal_here(dir: &str) -> Option<OpenApp> {
    let table: &[(&str, &[&str])] = &[
        ("alacritty", &["--working-directory"]),
        ("kitty", &["--directory"]),
        ("foot", &["--working-directory"]),
        ("wezterm", &["start", "--cwd"]),
        ("gnome-terminal", &["--working-directory"]),
        ("konsole", &["--workdir"]),
    ];
    // Ghostty uses a single `--working-directory=DIR` token.
    if on_path("ghostty") {
        return Some(OpenApp {
            label: "Terminal here".to_string(),
            kind: AppKind::Terminal,
            program: "ghostty".to_string(),
            args: vec![format!("--working-directory={dir}")],
            terminal: false,
        });
    }
    for &(bin, flags) in table {
        if on_path(bin) {
            let mut args: Vec<String> = flags.iter().map(|s| s.to_string()).collect();
            args.push(dir.to_string());
            return Some(OpenApp {
                label: "Terminal here".to_string(),
                kind: AppKind::Terminal,
                program: bin.to_string(),
                args,
                terminal: false,
            });
        }
    }
    None
}

/// Whether `bin` resolves to an executable file on `PATH`.
fn on_path(bin: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|dir| dir.join(bin).is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn candidates_are_well_formed() {
        let apps = candidates_for(&PathBuf::from("/tmp/notes.txt"), FileType::Text);
        for a in &apps {
            assert!(!a.program.is_empty());
            // The target path (or dir) is always baked into the args.
            assert!(!a.args.is_empty());
        }
    }

    #[test]
    fn media_extension_selects_media_first() {
        assert!(matches!(
            primary_category(FileType::Binary, "mp4"),
            Some(Cat::Media)
        ));
        assert!(primary_category(FileType::Binary, "o").is_none());
    }

    #[test]
    fn image_and_pdf_prefer_their_viewers() {
        assert!(matches!(
            primary_category(FileType::Image, "png"),
            Some(Cat::Image)
        ));
        assert!(matches!(
            primary_category(FileType::Pdf, "pdf"),
            Some(Cat::Pdf)
        ));
    }

    #[test]
    fn docx_has_no_primary_viewer() {
        // Word docs go to the OS default rather than a code editor.
        assert!(primary_category(FileType::Docx, "docx").is_none());
    }
}
