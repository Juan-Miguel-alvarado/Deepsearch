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
        // Generous: the very first request also loads the model into memory,
        // which can take a while on a cold/busy machine.
        .timeout(Duration::from_secs(120))
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
        "You turn a user's natural-language file-search request into a one-line \
query for a tool called `deepsearch`. Output ONLY the query — no quotes, no code \
fences, no explanation.\n\
\n\
Pieces you may use:\n\
- keywords: words that should appear in the file name or contents\n\
- a type filter, ONLY one of exactly: type:image type:pdf type:docx type:text type:binary\n\
- ext:<extension>  (e.g. ext:rs, ext:png; repeatable)\n\
\n\
Rules:\n\
- Use ONLY words from the user's request. NEVER invent words, and never copy \
words from the examples below.\n\
- Add a type:/ext: filter only when the request clearly implies a file kind or \
extension.\n\
- `type:` must be one of the five values listed above. For a programming \
language use its extension instead (rust->ext:rs, python->ext:py, \
javascript->ext:js, go->ext:go) — never write type:rust or type:code.\n\
- If the request names only a kind of file with no topic, output just the filter \
(e.g. `type:image`).\n\
- Keep it short; translate the keywords to English.\n\
\n\
Examples (patterns only — do NOT reuse their words):\n\
\"todas mis fotos\" -> type:image\n\
\"el contrato en pdf\" -> contract type:pdf\n\
\"scripts de backup en python\" -> backup ext:py\n\
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

/// Default embedding model (small, local, good for retrieval).
const DEFAULT_EMBED_MODEL: &str = "nomic-embed-text";

fn embed_model() -> String {
    match std::env::var("DEEPSEARCH_OLLAMA_EMBED_MODEL") {
        Ok(m) if !m.trim().is_empty() => m.trim().to_string(),
        _ => DEFAULT_EMBED_MODEL.to_string(),
    }
}

/// Whether the embedding model is installed in the local Ollama.
pub fn embed_available() -> bool {
    let want = embed_model();
    installed_models()
        .map(|ms| {
            ms.iter()
                .any(|m| m == &want || m.split(':').next() == Some(want.as_str()))
        })
        .unwrap_or(false)
}

/// A hint shown to the user when embeddings are needed but unavailable.
pub fn embed_setup_hint() -> String {
    format!(
        "semantic search needs a local embedding model. Install Ollama \
(https://ollama.com) and run `ollama pull {}`.",
        embed_model()
    )
}

/// Embed `text` into a unit-normalized vector via Ollama. `is_query` selects the
/// task prefix nomic-style models expect (queries and documents share the space
/// but are prefixed differently).
pub fn embed(text: &str, is_query: bool) -> Result<Vec<f32>, String> {
    let model = embed_model();
    let prefix = if is_query {
        "search_query: "
    } else {
        "search_document: "
    };
    let payload = serde_json::json!({
        "model": model,
        "prompt": format!("{prefix}{}", cap_text(text)),
    });
    let resp = gen_agent()
        .post(&format!("{}/api/embeddings", host()))
        .send_json(payload)
        .map_err(|e| format!("embedding request failed: {e}"))?;
    let body: serde_json::Value = resp
        .into_json()
        .map_err(|e| format!("bad embedding response: {e}"))?;
    let arr = body
        .get("embedding")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("no embedding returned. {}", embed_setup_hint()))?;
    let mut v: Vec<f32> = arr
        .iter()
        .filter_map(|x| x.as_f64().map(|f| f as f32))
        .collect();
    if v.is_empty() {
        return Err("empty embedding returned".to_string());
    }
    normalize(&mut v);
    Ok(v)
}

/// Unit-normalize in place so downstream ranking can use a plain dot product.
fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cap the text we embed: models have a context limit and very long documents
/// add latency without improving retrieval much.
fn cap_text(text: &str) -> String {
    const MAX_CHARS: usize = 8000;
    if text.chars().count() <= MAX_CHARS {
        text.to_string()
    } else {
        text.chars().take(MAX_CHARS).collect()
    }
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
