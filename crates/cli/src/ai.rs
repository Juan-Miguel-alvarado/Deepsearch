//! Optional natural-language search, powered by a **local** [Ollama](https://ollama.com)
//! server.
//!
//! This is entirely optional and free: if Ollama isn't installed/running the
//! feature simply isn't offered, and deepsearch works exactly as before. When it
//! is available, a plain-language request ("screenshots from my rust project")
//! is translated by the local model into deepsearch's own query syntax
//! (search terms plus `type:`/`ext:` filters) and run through the normal ranker.
//!
//! Nothing leaves the machine — the request goes to `localhost:11434`.

use std::time::Duration;

/// Default Ollama endpoint.
const DEFAULT_HOST: &str = "http://localhost:11434";

/// Base URL of the Ollama server. Respects Ollama's own `OLLAMA_HOST`, and the
/// model can be pinned with `DEEPSEARCH_OLLAMA_MODEL`.
fn host() -> String {
    match std::env::var("OLLAMA_HOST") {
        Ok(h) if !h.trim().is_empty() => {
            let h = h.trim();
            if h.starts_with("http://") || h.starts_with("https://") {
                h.to_string()
            } else {
                format!("http://{h}")
            }
        }
        _ => DEFAULT_HOST.to_string(),
    }
}

fn quick_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_millis(500))
        .timeout(Duration::from_secs(3))
        .build()
}

fn gen_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_millis(800))
        .timeout(Duration::from_secs(60))
        .build()
}

/// Whether a local Ollama server is reachable right now.
pub fn available() -> bool {
    quick_agent()
        .get(&format!("{}/api/tags", host()))
        .call()
        .is_ok()
}

/// Names of the models installed in the local Ollama.
fn installed_models() -> Result<Vec<String>, String> {
    let resp = quick_agent()
        .get(&format!("{}/api/tags", host()))
        .call()
        .map_err(|e| format!("cannot reach Ollama at {}: {e}", host()))?;
    let body: serde_json::Value = resp
        .into_json()
        .map_err(|e| format!("bad response from Ollama: {e}"))?;
    let models = body
        .get("models")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok(models)
}

/// Choose which model to use: `DEEPSEARCH_OLLAMA_MODEL` if set, else the first
/// model installed locally.
fn pick_model() -> Result<String, String> {
    if let Ok(m) = std::env::var("DEEPSEARCH_OLLAMA_MODEL") {
        if !m.trim().is_empty() {
            return Ok(m.trim().to_string());
        }
    }
    let models = installed_models()?;
    models.into_iter().next().ok_or_else(|| {
        "Ollama is running but has no models. Pull one, e.g. `ollama pull llama3.2`.".to_string()
    })
}

/// The instruction we give the model. It must answer with a single query line.
fn build_prompt(request: &str) -> String {
    format!(
        "You convert a user's natural-language file-search request into a query \
for a tool called `deepsearch`. Reply with ONLY the query line — no quotes, no \
code fences, no explanation.\n\
\n\
Query syntax:\n\
- plain words: matched against file names and contents (use meaningful keywords)\n\
- type:image | type:pdf | type:code | type:text | type:docx | type:binary  (filter by kind)\n\
- ext:rs ext:png ...  (filter by file extension; repeatable)\n\
\n\
Rules: keep it short; prefer English keywords; only add a type:/ext: filter if \
the request clearly implies one; if the request is already keywords, return them \
as-is.\n\
\n\
Examples:\n\
Request: rust files about parsing\n\
Query: parsing ext:rs\n\
Request: screenshots and photos\n\
Query: type:image\n\
Request: that invoice pdf from acme\n\
Query: invoice acme type:pdf\n\
Request: my notes on the database migration\n\
Query: database migration notes\n\
\n\
Request: {request}\n\
Query:"
    )
}

/// Translate a natural-language `request` into a deepsearch query string using
/// the local model. Returns a ready-to-run query, or a human-readable error.
pub fn translate_query(request: &str) -> Result<String, String> {
    let request = request.trim();
    if request.is_empty() {
        return Err("nothing to translate".to_string());
    }
    let model = pick_model()?;
    let payload = serde_json::json!({
        "model": model,
        "prompt": build_prompt(request),
        "stream": false,
        "options": { "temperature": 0 },
    });
    let resp = gen_agent()
        .post(&format!("{}/api/generate", host()))
        .send_json(payload)
        .map_err(|e| format!("Ollama request failed: {e}"))?;
    let body: serde_json::Value = resp
        .into_json()
        .map_err(|e| format!("bad response from Ollama: {e}"))?;
    let raw = body
        .get("response")
        .and_then(|r| r.as_str())
        .unwrap_or_default();
    let query = sanitize(raw);
    if query.is_empty() {
        return Err("the model returned an empty query".to_string());
    }
    Ok(query)
}

/// Clean the model's reply down to a single usable query line: take the first
/// non-empty line, drop a leading "Query:" echo and any surrounding quotes or
/// code fences.
fn sanitize(raw: &str) -> String {
    let line = raw
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let mut s = line.trim();
    // Drop a leading "Query:" the model sometimes echoes.
    if let Some(rest) = s
        .strip_prefix("Query:")
        .or_else(|| s.strip_prefix("query:"))
    {
        s = rest.trim();
    }
    // Strip surrounding backticks or quotes.
    s = s
        .trim_matches('`')
        .trim_matches('"')
        .trim_matches('\'')
        .trim();
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_takes_first_line_and_strips_prefix() {
        assert_eq!(sanitize("Query: parsing ext:rs"), "parsing ext:rs");
        assert_eq!(
            sanitize("  invoice type:pdf  \n\nmore text"),
            "invoice type:pdf"
        );
    }

    #[test]
    fn sanitize_strips_fences_and_quotes() {
        assert_eq!(sanitize("`type:image`"), "type:image");
        assert_eq!(sanitize("\"database migration\""), "database migration");
    }

    #[test]
    fn sanitize_empty() {
        assert_eq!(sanitize("\n\n  \n"), "");
    }
}
