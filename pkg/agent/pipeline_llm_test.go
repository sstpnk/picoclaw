package agent

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
)

func TestSemanticRetryError_EmptyContentRetriesByDefault(t *testing.T) {
	defaults := config.AgentDefaults{
		LLMRetryOnEmptyContent:  true,
		LLMRetryOnFinishReasons: []string{"length"},
	}
	resp := &providers.LLMResponse{Content: "", FinishReason: "stop"}

	err := semanticRetryError(defaults, resp, 0, 3)
	if err == nil {
		t.Fatal("expected retriable error for empty content, got nil")
	}
	if !errors.Is(err, errSemanticRetry) {
		t.Fatalf("expected error to wrap errSemanticRetry, got %v", err)
	}
	if !strings.Contains(err.Error(), "empty LLM response") {
		t.Fatalf("expected error to mention empty content, got %q", err.Error())
	}
}

func TestSemanticRetryError_WhitespaceOnlyContentRetries(t *testing.T) {
	defaults := config.AgentDefaults{LLMRetryOnEmptyContent: true}
	resp := &providers.LLMResponse{Content: "   \n\t  ", FinishReason: "stop"}

	if err := semanticRetryError(defaults, resp, 0, 3); err == nil {
		t.Fatal("expected retriable error for whitespace-only content, got nil")
	}
}

func TestSemanticRetryError_EmptyContentDisabled(t *testing.T) {
	defaults := config.AgentDefaults{LLMRetryOnEmptyContent: false}
	resp := &providers.LLMResponse{Content: "", FinishReason: "stop"}

	if err := semanticRetryError(defaults, resp, 0, 3); err != nil {
		t.Fatalf("expected nil when empty-content retry disabled, got %v", err)
	}
}

func TestSemanticRetryError_FinishReasonLengthRetries(t *testing.T) {
	defaults := config.AgentDefaults{
		LLMRetryOnEmptyContent:  true,
		LLMRetryOnFinishReasons: []string{"length"},
	}
	resp := &providers.LLMResponse{Content: "partial answer", FinishReason: "length"}

	err := semanticRetryError(defaults, resp, 0, 3)
	if err == nil {
		t.Fatal("expected retriable error for finish_reason=length, got nil")
	}
	if !strings.Contains(err.Error(), "length") {
		t.Fatalf("expected error to mention finish_reason, got %q", err.Error())
	}
}

func TestSemanticRetryError_FinishReasonStopDoesNotRetry(t *testing.T) {
	defaults := config.AgentDefaults{
		LLMRetryOnEmptyContent:  true,
		LLMRetryOnFinishReasons: []string{"length"},
	}
	resp := &providers.LLMResponse{Content: "ok", FinishReason: "stop"}

	if err := semanticRetryError(defaults, resp, 0, 3); err != nil {
		t.Fatalf("expected nil for finish_reason=stop, got %v", err)
	}
}

func TestSemanticRetryError_FinishReasonLengthButListEmpty(t *testing.T) {
	defaults := config.AgentDefaults{
		LLMRetryOnEmptyContent:  true,
		LLMRetryOnFinishReasons: nil,
	}
	resp := &providers.LLMResponse{Content: "ok", FinishReason: "length"}

	if err := semanticRetryError(defaults, resp, 0, 3); err != nil {
		t.Fatalf("expected nil when LLMRetryOnFinishReasons is empty, got %v", err)
	}
}

func TestSemanticRetryError_NilResponseDoesNotPanic(t *testing.T) {
	defaults := config.AgentDefaults{LLMRetryOnEmptyContent: true}
	if err := semanticRetryError(defaults, nil, 0, 3); err != nil {
		t.Fatalf("expected nil for nil response, got %v", err)
	}
}

func TestSemanticRetryError_TokenCapStyleFinishReason(t *testing.T) {
	// Some providers emit "max_tokens" or "model_length" instead of "length".
	// Operators can configure any of these.
	defaults := config.AgentDefaults{
		LLMRetryOnFinishReasons: []string{"length", "max_tokens"},
	}

	cases := []struct {
		reason string
		want   bool
	}{
		{"length", true},
		{"max_tokens", true},
		{"stop", false},
		{"content_filter", false},
	}

	for _, tc := range cases {
		resp := &providers.LLMResponse{Content: "ok", FinishReason: tc.reason}
		err := semanticRetryError(defaults, resp, 0, 3)
		got := err != nil
		if got != tc.want {
			t.Errorf("finish_reason=%q: retry=%v, want %v (err=%v)", tc.reason, got, tc.want, err)
		}
	}
}

func TestTransientLLMRetryReason_ClassifiesSemanticError(t *testing.T) {
	reason, ok := transientLLMRetryReason(fmt.Errorf("wrap: %w", errSemanticRetry))
	if !ok {
		t.Fatal("expected semantic error to be retriable")
	}
	if reason != "semantic" {
		t.Fatalf("expected reason %q, got %q", "semantic", reason)
	}
}

func TestTransientLLMRetryReason_NilError(t *testing.T) {
	reason, ok := transientLLMRetryReason(nil)
	if ok {
		t.Fatalf("expected nil error to be non-retriable, got reason=%q", reason)
	}
}
