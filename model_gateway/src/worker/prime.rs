//! Prefix-cache conditioning: one completion to every cache-owning worker.
//!
//! `POST /prime_prefix_cache` exists so a benchmark can start with the same
//! prefix already resident in every worker's KV cache. Normal routing
//! load-balances to exactly one worker, which is precisely what this must not
//! do, so this is a fan-out over the registry (the `flush_cache_all` shape)
//! rather than a routing decision. Nothing here touches `WorkerSelectionStage`,
//! `PolicyRegistry`, or any `RouterTrait`.
//!
//! The per-worker send is a *serving* request, not an admin RPC, so unlike
//! [`Worker::flush_cache`] it needs something from its caller: pre-encoded
//! `token_ids`. The token-only wires carry `TokenizedInput { original_text,
//! input_ids }` and the worker layer has no tokenizer at all -- the registry
//! lives on `AppContext`. Encoding once at the fan-out site is also the right
//! thing for this operation, since every target primes the *identical* prefix.
//!
//! ## Why this is not a `Worker` trait method
//!
//! The other admin ops (`flush_cache`, `start_profile`, `stop_profile`) are
//! trait defaults that dispatch on `connection_mode()`. This one cannot be,
//! for a concrete reason: the HTTP arm reuses [`serialize_request_body`], which
//! takes `&dyn Worker`, and a trait default body cannot coerce `&self` to
//! `&dyn Worker` (that requires `Self: Sized`). Reimplementing that body
//! construction inline would fork the single owner of the alias-to-canonical
//! `model` rewrite and -- the part that actually matters -- of
//! [`Worker::prepare_request`], which injects `data_parallel_rank` and is the
//! only reason per-rank coverage is real over HTTP. A free function taking
//! `&Arc<dyn Worker>` gets `&dyn Worker` for free and keeps this file's
//! streaming lifecycle out of `worker.rs`.

use std::{sync::Arc, time::Duration};

use http::header::CONTENT_TYPE;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::completion::CompletionRequest;
use uuid::Uuid;

use crate::{
    routers::{
        grpc::utils::tonic_ext::{TonicResultExt, TonicStatusExt},
        http::request_body::{serialize_request_body, RequestBodyError},
    },
    worker::{ConnectionMode, Worker},
};

/// The engine route a prime is sent to: the same one the HTTP proxy uses for
/// normal completions serving.
const COMPLETIONS_ROUTE: &str = "/v1/completions";

/// Bound on one prime send.
///
/// A prime is a full prefill of the benchmark's shared prefix, so it is
/// bounded far more loosely than the unary admin ops (`FLUSH_HTTP_TIMEOUT` is
/// 45s): paying that prefill once, up front, is the entire point of the
/// endpoint. A bound is mandatory rather than defensive --
/// `BackendClient::generate` has no deadline of its own, and an admin fan-out
/// has no client disconnect to fall back on, so a wedged engine would
/// otherwise hang the whole request forever.
pub const PRIME_TIMEOUT: Duration = Duration::from_secs(300);

/// Everything one prime send needs that a [`Worker`] cannot compute itself.
pub struct PrimeRequest<'a> {
    /// The normalized serving body: canonical `model`, `stream` forced false,
    /// `rid` cleared. Built once by the handler and shared by every target.
    pub body: &'a CompletionRequest,
    /// The shared prefix, verbatim. Rides `TokenizedInput.original_text`
    /// (cosmetic, for worker logs) on the token-only wires.
    pub prompt: &'a str,
    /// `prompt` encoded with `add_special_tokens = false`, the same flag
    /// `CompletionPreparationStage` passes. Encoding it any other way primes a
    /// different token sequence than the benchmark then sends: every target
    /// would report 200 on a silent no-op.
    pub token_ids: &'a [u32],
    /// Handed to `BackendClient::finalize_generate_request`. `None` on a
    /// pure-HTTP deployment, which may legitimately have no tokenizer
    /// registered, since the registry is populated from gRPC workers.
    pub tokenizer: Option<&'a Arc<dyn Tokenizer>>,
    /// Canonical model id when the caller used an alias, for the HTTP body
    /// rewrite. `None` when `body.model` is already canonical.
    pub canonical_model: Option<&'a str>,
    pub timeout: Duration,
}

/// Outcome of one prime send. Infallible by shape: the contract reports one row
/// per target, so a failure is data, not an error.
pub struct PrimeOutcome {
    pub http_status: Option<u16>,
    pub error: Option<String>,
}

impl PrimeOutcome {
    fn ok(status: u16) -> Self {
        Self {
            http_status: Some(status),
            error: None,
        }
    }

    fn failed(status: Option<u16>, error: impl Into<String>) -> Self {
        Self {
            http_status: status,
            error: Some(error.into()),
        }
    }
}

/// Send one conditioning completion to a single named worker.
///
/// Per-transport dispatch mirrors [`Worker::flush_cache`].
pub(crate) async fn prime_worker(
    worker: &Arc<dyn Worker>,
    req: &PrimeRequest<'_>,
) -> PrimeOutcome {
    match tokio::time::timeout(req.timeout, prime_once(worker, req)).await {
        Ok(outcome) => outcome,
        // Dropping the in-flight future drops any live `ProtoStream`, and
        // `AbortOnDropStream::drop` then sends the Abort that tears the
        // abandoned generation down on the engine, which is exactly what we
        // want for a prime we have given up on.
        Err(_) => PrimeOutcome::failed(
            None,
            format!("prime timed out after {:?}", req.timeout),
        ),
    }
}

async fn prime_once(worker: &Arc<dyn Worker>, req: &PrimeRequest<'_>) -> PrimeOutcome {
    match worker.connection_mode() {
        ConnectionMode::Http => prime_http(worker, req).await,
        // ZMQ is deliberately NOT rejected up front the way `flush_cache`
        // rejects it. That guard exists because EngineCore exposes no admin
        // RPCs; generate is a first-class ZMQ operation, and
        // `BasicWorker::get_backend_client`'s ZMQ arm peeks and fails fast
        // rather than driving the lazy handshake, so a ZMQ worker either
        // primes or reports that its handshake has not completed. It never
        // stalls the fan-out.
        ConnectionMode::Grpc | ConnectionMode::Zmq => prime_backend(worker, req).await,
    }
}

fn is_success(status: u16) -> bool {
    (200..300).contains(&status)
}

/// HTTP engines tokenize the raw prompt themselves, so `req.token_ids` is
/// unused on this path.
async fn prime_http(worker: &Arc<dyn Worker>, req: &PrimeRequest<'_>) -> PrimeOutcome {
    // The single owner of outbound body construction: alias-to-canonical
    // `model`, `Worker::prepare_request` (which injects `data_parallel_rank`
    // for a rank-expanded worker -- this is what makes per-rank coverage real
    // over HTTP), and the SGLang default-field strip. The flattened
    // `CompletionRequest.other` survives it, so the bench's extra fields do
    // literally pass through on this transport.
    let body = match serialize_request_body(req.body, req.canonical_model, worker.as_ref()) {
        Ok(body) => body,
        Err(RequestBodyError::Serialize(e)) => {
            return PrimeOutcome::failed(None, format!("failed to serialize prime body: {e}"))
        }
        Err(RequestBodyError::Prepare(e)) => {
            return PrimeOutcome::failed(None, format!("failed to prepare prime body: {e}"))
        }
    };

    // The per-worker client carries a default request timeout, which for a long
    // prefill would fire long before `PRIME_TIMEOUT`. Override it, exactly as
    // `admin_http_post` does for the other admin ops.
    let mut builder = worker
        .http_client()
        .post(worker.endpoint_url(COMPLETIONS_ROUTE))
        .header(CONTENT_TYPE, "application/json")
        .timeout(req.timeout)
        .body(body);
    if let Some(key) = worker.api_key() {
        builder = builder.bearer_auth(key);
    }

    match builder.send().await {
        Ok(response) => {
            let status = response.status().as_u16();
            // Same circuit-breaker signal a served request records.
            worker.record_outcome(status);
            // Read and discard: the completion text is irrelevant, but leaving
            // the body unread returns a dirty connection to the pool.
            let _ = response.bytes().await;
            if is_success(status) {
                PrimeOutcome::ok(status)
            } else {
                PrimeOutcome::failed(Some(status), format!("HTTP {status}"))
            }
        }
        // No response at all: transport error or reqwest's own timeout.
        Err(e) => PrimeOutcome::failed(None, e.to_string()),
    }
}

/// The token-only wires: gRPC (any engine) and direct ZMQ, both behind
/// [`crate::routers::grpc::backend_client::BackendClient`].
async fn prime_backend(worker: &Arc<dyn Worker>, req: &PrimeRequest<'_>) -> PrimeOutcome {
    let client = match worker.get_backend_client().await {
        Ok(Some(client)) => client,
        Ok(None) => return PrimeOutcome::failed(None, "worker has no backend client"),
        Err(e) => return PrimeOutcome::failed(None, e.to_string()),
    };
    // `BackendClient::generate` takes `&mut self`; the pipeline's client
    // acquisition stage clones out of the cached `Arc` exactly this way.
    let mut client = (*client).clone();

    let request_id = format!("prime-{}", Uuid::now_v7());

    // Byte for byte the call the serving pipeline makes, so all
    // TokenSpeed/SGLang/vLLM/TRT-LLM/MLX dispatch, `TokenizedInput` shaping and
    // sampling-param mapping stay in one place.
    let mut proto = match client.build_completion_request(
        request_id,
        req.body,
        req.prompt.to_string(),
        req.token_ids.to_vec(),
    ) {
        Ok(proto) => proto,
        Err(e) => {
            return PrimeOutcome::failed(None, format!("failed to build prime request: {e}"))
        }
    };

    // A no-op for a gRPC TokenSpeed worker -- `resolve_string_stops`' TokenSpeed
    // arm is gated on `token_only_wire` and `is_zmq()` is false -- but
    // load-bearing for SGLang-gRPC and for every ZMQ backend, which cannot match
    // string stops themselves. The returned residual stops are the router's
    // trimming obligation for output text; a prime discards output.
    let _router_stops = client.finalize_generate_request(&mut proto, req.tokenizer);

    // Mirrors `RequestExecutionStage::execute_single`. Silently a no-op for
    // TokenSpeed: `ProtoGenerateRequest::set_data_parallel_rank`'s
    // `Trtllm | Mlx | TokenSpeed` arm is empty and `GenerateRequest` has no rank
    // field. That is exactly why a TokenSpeed target reports rank 0 -- the
    // control plane's coverage check then fails loudly for DP > 1 instead of
    // priming one rank N times.
    if let Some(rank) = worker.dp_rank() {
        proto.set_data_parallel_rank(rank as i32);
    }

    let result = client.generate(proto).await;
    // Same circuit-breaker signal `execute_single` records. A prime is a real
    // inference request: an RPC that fails here means the worker is genuinely
    // sick.
    worker.record_outcome(result.cb_status_code());

    let mut stream = match result {
        Ok(stream) => stream,
        Err(status) => {
            return PrimeOutcome::failed(
                Some(status.http_status().as_u16()),
                format!("generate failed: {}", status.message()),
            )
        }
    };

    // Drain to end of stream. The prime is complete only when the engine has
    // finished the request, and only then has its prefix cache been written;
    // returning at the first chunk would report success before the prefill
    // landed. This is `collect_stream_responses` with the `Complete`/`Chunk`
    // classification deleted, since `max_tokens: 1` output is discarded and only
    // success matters.
    while let Some(item) = stream.next().await {
        if let Err(status) = item {
            // Left unmarked on purpose so `Drop` sends the Abort, exactly as
            // `collect_stream_responses` does for the error case.
            return PrimeOutcome::failed(
                Some(status.http_status().as_u16()),
                format!("prime stream failed: {}", status.message()),
            );
        }
    }
    // Mandatory: without it `AbortOnDropStream::drop` fires an Abort RPC,
    // tearing down the very request whose prefill was meant to fill the cache.
    stream.mark_completed();

    // A stream that ran to completion is the backend equivalent of 2xx.
    // Reporting it as 200 is what makes a successful gRPC prime unambiguously
    // pass the control plane's "every target's http_status is 2xx" rule.
    PrimeOutcome::ok(200)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use http::StatusCode;
    use openai_protocol::worker::WorkerType;

    use super::*;
    use crate::worker::{manager::WorkerManager, BasicWorkerBuilder};

    /// A one-route stand-in for an engine's `/v1/completions`: records every
    /// body it receives and answers with the configured status after `delay`.
    #[expect(
        clippy::disallowed_methods,
        reason = "test-local server task; the test process owns its lifetime"
    )]
    async fn spawn_engine(
        status: StatusCode,
        delay: Duration,
    ) -> (String, Arc<Mutex<Vec<serde_json::Value>>>) {
        let seen: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);

        let app = axum::Router::new().route(
            "/v1/completions",
            axum::routing::post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                let recorder = Arc::clone(&recorder);
                async move {
                    recorder.lock().unwrap().push(body);
                    if !delay.is_zero() {
                        tokio::time::sleep(delay).await;
                    }
                    (status, axum::Json(serde_json::json!({ "choices": [] })))
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), seen)
    }

    fn http_worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .build(),
        )
    }

    fn body(prompt: &str) -> CompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "m",
            "prompt": prompt,
            "stream": false,
            "n": 1,
            "max_tokens": 1,
            "temperature": 0.0,
            "some_bench_field": "carried"
        }))
        .unwrap()
    }

    fn request<'a>(
        body: &'a CompletionRequest,
        prompt: &'a str,
        token_ids: &'a [u32],
        timeout: Duration,
    ) -> PrimeRequest<'a> {
        PrimeRequest {
            body,
            prompt,
            token_ids,
            tokenizer: None,
            canonical_model: None,
            timeout,
        }
    }

    /// The whole point of the endpoint: EVERY target gets exactly one request
    /// and each reports its own row. A regression to normal load balancing
    /// would send one request in total and produce one target.
    #[tokio::test]
    async fn fan_out_sends_exactly_one_request_to_every_target() {
        let (url_a, seen_a) = spawn_engine(StatusCode::OK, Duration::ZERO).await;
        let (url_b, seen_b) = spawn_engine(StatusCode::OK, Duration::ZERO).await;
        let workers = vec![http_worker(&url_a), http_worker(&url_b)];

        let body = body("shared prefix");
        let result = WorkerManager::prime_prefix_cache_all(
            workers,
            &request(&body, "shared prefix", &[], Duration::from_secs(5)),
        )
        .await;

        assert_eq!(result.targets.len(), 2);
        assert_eq!(seen_a.lock().unwrap().len(), 1);
        assert_eq!(seen_b.lock().unwrap().len(), 1);

        let mut urls: Vec<&str> = result.targets.iter().map(|t| t.url.as_str()).collect();
        urls.sort_unstable();
        let mut expected = vec![url_a.as_str(), url_b.as_str()];
        expected.sort_unstable();
        assert_eq!(urls, expected);

        for target in &result.targets {
            assert_eq!(target.http_status, Some(200));
            assert!(target.error.is_none());
            assert_eq!(target.rank, 0);
        }
    }

    /// The body the engine receives must be a normal serving request: the
    /// prompt verbatim, `stream` false (a streamed prime would report 200
    /// before the prefill finished), and the benchmark's extra fields carried
    /// through `CompletionRequest.other`.
    #[tokio::test]
    async fn http_prime_forwards_a_normal_non_streaming_serving_body() {
        let (url, seen) = spawn_engine(StatusCode::OK, Duration::ZERO).await;
        let body = body("shared prefix");

        let outcome = prime_worker(
            &http_worker(&url),
            &request(&body, "shared prefix", &[], Duration::from_secs(5)),
        )
        .await;

        assert_eq!(outcome.http_status, Some(200));
        assert!(outcome.error.is_none());

        let captured = seen.lock().unwrap();
        let sent = &captured[0];
        assert_eq!(sent["prompt"], "shared prefix");
        assert_eq!(sent["stream"], false);
        assert_eq!(sent["max_tokens"], 1);
        assert_eq!(sent["some_bench_field"], "carried");
        assert!(sent.get("data_parallel_rank").is_none());
    }

    /// A non-2xx engine answer is data, not an error: report the number so the
    /// control plane can apply its own 2xx rule.
    #[tokio::test]
    async fn http_prime_reports_the_engine_status_on_failure() {
        let (url, _seen) = spawn_engine(StatusCode::SERVICE_UNAVAILABLE, Duration::ZERO).await;
        let body = body("p");

        let outcome = prime_worker(
            &http_worker(&url),
            &request(&body, "p", &[], Duration::from_secs(5)),
        )
        .await;

        assert_eq!(outcome.http_status, Some(503));
        assert!(outcome.error.is_some());
    }

    /// A wedged engine must not hang the fan-out. Without the deadline this
    /// test never returns.
    #[tokio::test]
    async fn a_hung_target_times_out_with_a_null_status() {
        let (url, _seen) = spawn_engine(StatusCode::OK, Duration::from_secs(30)).await;
        let body = body("p");

        let outcome = prime_worker(
            &http_worker(&url),
            &request(&body, "p", &[], Duration::from_millis(150)),
        )
        .await;

        assert!(outcome.http_status.is_none());
        assert!(outcome.error.is_some());
    }

    /// No response at all (nothing listening) reports a null status rather
    /// than inventing a synthetic one.
    #[tokio::test]
    async fn an_unreachable_target_reports_a_null_status() {
        let body = body("p");

        let outcome = prime_worker(
            // Port 1 is reserved and never listening.
            &http_worker("http://127.0.0.1:1"),
            &request(&body, "p", &[], Duration::from_secs(2)),
        )
        .await;

        assert!(outcome.http_status.is_none());
        assert!(outcome.error.is_some());
    }

    /// Per-rank coverage over HTTP: all ranks of one replica share a base url
    /// (which is what the control plane groups on) and differ only by the
    /// `data_parallel_rank` that `Worker::prepare_request` injects. This is the
    /// behavior that would be lost by hand-rolling the body instead of going
    /// through `serialize_request_body`.
    #[tokio::test]
    async fn dp_ranks_share_a_base_url_and_carry_their_rank_into_the_body() {
        let (url, seen) = spawn_engine(StatusCode::OK, Duration::ZERO).await;
        let workers: Vec<Arc<dyn Worker>> = (0..2)
            .map(|rank| {
                Arc::new(
                    BasicWorkerBuilder::new(url.clone())
                        .worker_type(WorkerType::Regular)
                        .connection_mode(ConnectionMode::Http)
                        .dp_config(rank, 2)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();

        let body = body("p");
        let result = WorkerManager::prime_prefix_cache_all(
            workers,
            &request(&body, "p", &[], Duration::from_secs(5)),
        )
        .await;

        assert_eq!(result.targets.len(), 2);
        // `base_url()` strips the gateway-internal `@rank` suffix.
        assert!(result.targets.iter().all(|t| t.url == url));
        let mut ranks: Vec<usize> = result.targets.iter().map(|t| t.rank).collect();
        ranks.sort_unstable();
        assert_eq!(ranks, vec![0, 1]);

        let captured = seen.lock().unwrap();
        let mut sent_ranks: Vec<u64> = captured
            .iter()
            .map(|body| body["data_parallel_rank"].as_u64().unwrap())
            .collect();
        sent_ranks.sort_unstable();
        assert_eq!(sent_ranks, vec![0, 1]);
    }
}
