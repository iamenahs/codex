pub use codex_api::ResponseEvent;
use codex_api::create_text_param_for_request;
use codex_protocol::error::Result;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::ContextBaseline;
use codex_tools::ToolSpec;
use codex_tools::create_tools_json_for_responses_lite;
use codex_utils_output_truncation::approx_token_count;
use futures::Stream;
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// API request payload for a single model turn
#[derive(Debug, Clone)]
pub struct Prompt {
    /// Conversation context input items.
    pub input: Vec<ResponseItem>,

    /// Tools available to the model, including additional tools sourced from
    /// external MCP servers.
    pub(crate) tools: Arc<[ToolSpec]>,

    /// Whether parallel tool calls are permitted for this prompt.
    pub(crate) parallel_tool_calls: bool,

    pub base_instructions: BaseInstructions,

    /// Optional the output schema for the model's response.
    pub output_schema: Option<Value>,

    /// Whether the Responses API should strictly validate `output_schema`.
    pub output_schema_strict: bool,
}

impl Default for Prompt {
    fn default() -> Self {
        Self {
            input: Vec::new(),
            tools: Arc::default(),
            parallel_tool_calls: false,
            base_instructions: BaseInstructions::default(),
            output_schema: None,
            output_schema_strict: true,
        }
    }
}

/// How `build_responses_request` will serialize the tools for this request.
///
/// The two forms are not the same size: responses-lite against a provider that
/// namespaces tools folds every function and freeform spec into a single
/// namespace object, trading per-tool envelopes for one shared description.
/// Every other combination sends the specs one after another, which
/// `create_tools_json_for_responses_api` and
/// `create_tools_raw_json_for_responses_api` produce identical JSON for.
/// Pricing the tool surface in the wrong form misstates it by that difference,
/// so the caller passes the form the request will actually use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolsWireFormat {
    /// One namespace object; see `create_tools_json_for_responses_lite`.
    Namespaced,
    /// The specs verbatim, one after another.
    PerTool,
}

impl ToolsWireFormat {
    /// Mirrors the branch in `ModelClient::build_responses_request`.
    pub(crate) fn for_request(use_responses_lite: bool, namespace_tools: bool) -> Self {
        if use_responses_lite && namespace_tools {
            Self::Namespaced
        } else {
            Self::PerTool
        }
    }

    /// The tool payload as a string, or `None` if it will not serialize — in
    /// which case the request itself is about to fail, and guessing at a size
    /// would only make the failure quieter.
    fn render(self, tools: &[ToolSpec]) -> Option<String> {
        match self {
            Self::Namespaced => {
                serde_json::to_string(&create_tools_json_for_responses_lite(tools).ok()?).ok()
            }
            Self::PerTool => serde_json::to_string(tools).ok(),
        }
    }
}

impl Prompt {
    /// What this request costs before its conversation is counted: the base
    /// instructions, the tool payload, and the response format if there is one.
    ///
    /// A lower bound, not the whole fixed cost — see [`ContextBaseline`] for
    /// what is injected into the conversation and therefore invisible here.
    ///
    /// Priced with `approx_token_count`, the same byte-density heuristic the
    /// history estimator and the truncation helpers already use, so this is a
    /// figure of the same coarse kind — and the same kind the constant it
    /// replaces was standing in for. Every term is measured from the serialized
    /// payload rather than from the value behind it, because the payload is
    /// what is sent.
    pub(crate) fn context_baseline(&self, tools_wire_format: ToolsWireFormat) -> ContextBaseline {
        let tokens = |text: &str| i64::try_from(approx_token_count(text)).unwrap_or(i64::MAX);

        let tool_schema_tokens = tools_wire_format
            .render(&self.tools)
            .as_deref()
            .map_or(0, tokens);

        // The request does not carry the bare schema: `build_responses_request`
        // sends whatever `create_text_param_for_request` returns, which wraps it
        // in a format object with a type, a name and the strict flag. Price the
        // wrapper by building the same value.
        //
        // Verbosity is deliberately passed as `None`. It travels in the same
        // request field but it is not part of the response format, and folding
        // it in here would make a request that carries no schema at all report
        // a non-zero `output_schema_tokens`.
        let output_schema_tokens =
            create_text_param_for_request(None, &self.output_schema, self.output_schema_strict)
                .as_ref()
                .and_then(|text| serde_json::to_string(text).ok())
                .as_deref()
                .map_or(0, tokens);

        ContextBaseline {
            system_prompt_tokens: tokens(&self.base_instructions.text),
            tool_schema_tokens,
            output_schema_tokens,
            tool_count: i64::try_from(self.tools.len()).unwrap_or(i64::MAX),
        }
    }

    pub(crate) fn get_formatted_input_for_request(
        &self,
        use_responses_lite: bool,
    ) -> Vec<ResponseItem> {
        let mut input = self.input.clone();
        if use_responses_lite {
            strip_image_details(&mut input);
        }
        input
    }
}

fn strip_image_details(items: &mut [ResponseItem]) {
    for item in items {
        match item {
            ResponseItem::Message { content, .. } => {
                for content_item in content {
                    if let ContentItem::InputImage { detail, .. } = content_item {
                        *detail = None;
                    }
                }
            }
            ResponseItem::FunctionCallOutput { output, .. }
            | ResponseItem::CustomToolCallOutput { output, .. } => {
                if let Some(content) = output.content_items_mut() {
                    for content_item in content {
                        if let FunctionCallOutputContentItem::InputImage { detail, .. } =
                            content_item
                        {
                            *detail = None;
                        }
                    }
                }
            }
            ResponseItem::AdditionalTools { .. }
            | ResponseItem::Reasoning { .. }
            | ResponseItem::AgentMessage { .. }
            | ResponseItem::LocalShellCall { .. }
            | ResponseItem::FunctionCall { .. }
            | ResponseItem::ToolSearchCall { .. }
            | ResponseItem::CustomToolCall { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::ImageGenerationCall { .. }
            | ResponseItem::Compaction { .. }
            | ResponseItem::CompactionTrigger { .. }
            | ResponseItem::ContextCompaction { .. }
            | ResponseItem::Other => {}
        }
    }
}

pub struct ResponseStream {
    pub(crate) rx_event: mpsc::Receiver<Result<ResponseEvent>>,
    /// Signals the mapper task that the consumer stopped polling before the
    /// provider stream reached its own terminal event.
    pub(crate) consumer_dropped: CancellationToken,
}

impl Stream for ResponseStream {
    type Item = Result<ResponseEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx_event.poll_recv(cx)
    }
}

impl Drop for ResponseStream {
    fn drop(&mut self) {
        self.consumer_dropped.cancel();
    }
}

#[cfg(test)]
#[path = "client_common_tests.rs"]
mod tests;
