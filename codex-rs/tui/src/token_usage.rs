//! TUI token usage models and display formatting.

use std::fmt;

use codex_protocol::num_format::format_with_separators;
use codex_protocol::protocol::ContextBaseline as ProtocolContextBaseline;
use codex_protocol::protocol::context_baseline_tokens;
use codex_protocol::protocol::percent_of_context_window_remaining;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: i64,
    pub cached_input_tokens: i64,
    pub output_tokens: i64,
    pub reasoning_output_tokens: i64,
    pub total_tokens: i64,
}

impl TokenUsage {
    pub fn is_zero(&self) -> bool {
        self.total_tokens == 0
    }

    pub(crate) fn cached_input(&self) -> i64 {
        self.cached_input_tokens.max(0)
    }

    pub(crate) fn non_cached_input(&self) -> i64 {
        (self.input_tokens - self.cached_input()).max(0)
    }

    pub(crate) fn blended_total(&self) -> i64 {
        (self.non_cached_input() + self.output_tokens.max(0)).max(0)
    }

    /// Returns the raw `total_tokens` value. For `last_token_usage`, this is the latest active
    /// context size; for `total_token_usage`, this is the accumulated session total.
    pub(crate) fn tokens_in_context_window(&self) -> i64 {
        self.total_tokens
    }

    pub(crate) fn percent_of_context_window_remaining(
        &self,
        context_window: i64,
        baseline: i64,
    ) -> i64 {
        percent_of_context_window_remaining(
            self.tokens_in_context_window(),
            context_window,
            baseline,
        )
    }
}

/// The TUI's own serde-owned copy of `codex_protocol::protocol::ContextBaseline`.
///
/// The shape is duplicated because this crate persists its token models in its
/// own format; the arithmetic is not — everything that turns these numbers into
/// a percentage lives in `codex-protocol` and is called through
/// [`ProtocolContextBaseline`]. See there for what the figure covers and, more
/// importantly, what it does not.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ContextBaseline {
    pub(crate) system_prompt_tokens: i64,
    pub(crate) tool_schema_tokens: i64,
    pub(crate) output_schema_tokens: i64,
    pub(crate) tool_count: i64,
}

impl From<ContextBaseline> for ProtocolContextBaseline {
    fn from(value: ContextBaseline) -> Self {
        Self {
            system_prompt_tokens: value.system_prompt_tokens,
            tool_schema_tokens: value.tool_schema_tokens,
            output_schema_tokens: value.output_schema_tokens,
            tool_count: value.tool_count,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TokenUsageInfo {
    pub(crate) total_token_usage: TokenUsage,
    pub(crate) last_token_usage: TokenUsage,
    pub(crate) model_context_window: Option<i64>,
    #[serde(default)]
    pub(crate) context_baseline: Option<ContextBaseline>,
}

impl TokenUsageInfo {
    /// The baseline this session's percentage divides by, resolved by the same
    /// `codex-protocol` function the core uses.
    pub(crate) fn baseline_tokens(&self) -> i64 {
        context_baseline_tokens(self.context_baseline.map(Into::into))
    }

    /// Percent of the window the conversation can still grow into.
    pub(crate) fn percent_of_context_window_remaining(&self, context_window: i64) -> i64 {
        self.last_token_usage
            .percent_of_context_window_remaining(context_window, self.baseline_tokens())
    }
}

impl fmt::Display for TokenUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Token usage: total={} input={}{} output={}{}",
            format_with_separators(self.blended_total()),
            format_with_separators(self.non_cached_input()),
            if self.cached_input() > 0 {
                format!(
                    " (+ {} cached)",
                    format_with_separators(self.cached_input())
                )
            } else {
                String::new()
            },
            format_with_separators(self.output_tokens),
            if self.reasoning_output_tokens > 0 {
                format!(
                    " (reasoning {})",
                    format_with_separators(self.reasoning_output_tokens)
                )
            } else {
                String::new()
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::protocol::TokenUsage as ProtocolTokenUsage;
    use codex_protocol::protocol::TokenUsageInfo as ProtocolTokenUsageInfo;

    /// This crate keeps its own token models for its own persistence, so the
    /// shapes are duplicated on purpose. The arithmetic is not: both sides must
    /// resolve the same baseline and print the same percentage, or the status
    /// line and the app server disagree about the same session.
    #[test]
    fn the_percentage_matches_the_protocol_for_the_same_inputs() {
        let window = 272_000;
        let cases = [
            (0, None),
            (30_000, None),
            (141_000, None),
            (271_999, None),
            (30_000, Some((3_200, 1_800, 0, 6))),
            (30_000, Some((3_200, 28_000, 250, 96))),
            (141_000, Some((3_200, 28_000, 250, 96))),
            (271_999, Some((8_000, 300_000, 0, 400))),
        ];

        for (total_tokens, baseline) in cases {
            let baseline = baseline.map(|(prompt, tools, output, count)| ContextBaseline {
                system_prompt_tokens: prompt,
                tool_schema_tokens: tools,
                output_schema_tokens: output,
                tool_count: count,
            });
            let tui = TokenUsageInfo {
                last_token_usage: TokenUsage {
                    total_tokens,
                    ..TokenUsage::default()
                },
                model_context_window: Some(window),
                context_baseline: baseline,
                ..TokenUsageInfo::default()
            };
            let protocol = ProtocolTokenUsageInfo {
                last_token_usage: ProtocolTokenUsage {
                    total_tokens,
                    ..ProtocolTokenUsage::default()
                },
                model_context_window: Some(window),
                context_baseline: baseline.map(Into::into),
                ..ProtocolTokenUsageInfo::default()
            };

            assert_eq!(
                tui.baseline_tokens(),
                protocol.baseline_tokens(),
                "baseline drift for {total_tokens} / {baseline:?}"
            );
            assert_eq!(
                tui.percent_of_context_window_remaining(window),
                protocol.percent_of_context_window_remaining(window),
                "percentage drift for {total_tokens} / {baseline:?}"
            );
        }
    }
}
