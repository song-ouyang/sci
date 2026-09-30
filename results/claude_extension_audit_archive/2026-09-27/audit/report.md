# Claude extension provenance audit

**Status: blocked for primary comparison.**

Audited at: 2026-09-27T05:03:27.870408+00:00

Planned sessions: 18; structurally valid prediction arrays: 18; consistent current artifact chains: 15; eligible primary sessions: 0; sessions explicitly marked contaminated: 1.

All 18 archived sessions are excluded from the primary controlled comparison. Artifact consistency is separate from verified model inference. No answer-key dataset files or legacy scores were read; saved response or metadata content may itself contain prior label disclosures. No historical freeze, model identity or event trace is reconstructed. Matching outputs alone are not evidence of copying.

## Global limitations

- `HASH_MISSING_OR_INVALID`: protocol.parent_protocol_sha256
- `PRE_SCORE_FREEZE_MISSING`: No prediction frozen_manifest exists; the source manifest covers inputs only.
- `AGENT_TOOL_ISOLATION_UNVERIFIED`: The archived Agent-tool transport allowed tools; instruction-level prohibition does not establish tool or filesystem isolation.
- `NATIVE_EVENT_PROVENANCE_MISSING`: The extension has no captured provider-native request/response and tool-event chain authenticating its delivered context and completion.
- `MODEL_IDENTITY_UNVERIFIED`: Caller-written model names and a MODEL_ID self-report are not provider-returned model identity evidence.
- `ANSWER_KEY_ISOLATION_UNVERIFIED`: Current output hashes and answer_keys_read=false cannot demonstrate pre-score answer-key isolation.
- `SCHEDULE_MODEL_MISMATCH`: The session schedule does not enumerate the same 18 model/group identities as the extension protocol and output paths.

## Session inventory

| Requested model | Group | Array valid | Status | Eligible |
|---|---|---|---|---|
| claude-sonnet-4-6 | cmapss_clean_r1 | True | provenance_unverified | false |
| claude-sonnet-4-6 | cmapss_clean_r2 | True | provenance_unverified | false |
| claude-sonnet-4-6 | cmapss_clean_r3 | True | provenance_unverified | false |
| claude-sonnet-4-6 | tep_clean_r1 | True | provenance_unverified | false |
| claude-sonnet-4-6 | tep_clean_r2 | True | artifact_invalid | false |
| claude-sonnet-4-6 | tep_clean_r3 | True | artifact_invalid | false |
| claude-sonnet-4-6 | tep_misnamed_r1 | True | artifact_invalid | false |
| claude-sonnet-4-6 | tep_misnamed_r2 | True | provenance_unverified | false |
| claude-sonnet-4-6 | tep_misnamed_r3 | True | reported_contaminated | false |
| claude-opus-5-5 | cmapss_clean_r1 | True | provenance_unverified | false |
| claude-opus-5-5 | cmapss_clean_r2 | True | provenance_unverified | false |
| claude-opus-5-5 | cmapss_clean_r3 | True | provenance_unverified | false |
| claude-opus-5-5 | tep_clean_r1 | True | provenance_unverified | false |
| claude-opus-5-5 | tep_clean_r2 | True | provenance_unverified | false |
| claude-opus-5-5 | tep_clean_r3 | True | provenance_unverified | false |
| claude-opus-5-5 | tep_misnamed_r1 | True | provenance_unverified | false |
| claude-opus-5-5 | tep_misnamed_r2 | True | provenance_unverified | false |
| claude-opus-5-5 | tep_misnamed_r3 | True | provenance_unverified | false |

## Session-specific findings

- `claude-sonnet-4-6/tep_clean_r2`:
  - `HASH_MISMATCH`: metadata.prediction_sha256
  - `HASH_MISSING_OR_INVALID`: metadata.request_sha256
- `claude-sonnet-4-6/tep_clean_r3`:
  - `FILE_UNREADABLE`: extension:predictions/claude-sonnet-4-6/tep_clean_r3_request.json: [Errno 2] No such file or directory: '/Users/guoba/work/sci-paper-llm-pdm/results/revision_v3_claude_extension/predictions/claude-sonnet-4-6/tep_clean_r3_request.json'
  - `RAW_RESPONSE_INVALID`: Expecting value: line 1 column 1 (char 0)
  - `REQUEST_INVALID`: Missing or non-object request.
  - `HASH_MISSING_OR_INVALID`: metadata.prediction_sha256
  - `HASH_MISSING_OR_INVALID`: metadata.request_sha256
  - `HASH_MISSING_OR_INVALID`: metadata.response_text_sha256
  - `FIELD_MISMATCH`: metadata.model_requested differs from expected value
  - `FIELD_MISMATCH`: metadata.group_name differs from expected value
  - `FIELD_MISMATCH`: metadata.status differs from expected value
  - `FIELD_MISMATCH`: metadata.transport differs from expected value
  - `FIELD_MISMATCH`: metadata.answer_keys_read differs from expected value
  - `FIELD_MISMATCH`: metadata.dataset differs from expected value
  - `FIELD_MISMATCH`: metadata.condition differs from expected value
  - `FIELD_MISMATCH`: metadata.repeat differs from expected value
  - `FIELD_MISMATCH`: metadata.n_predictions differs from expected value
  - `FIELD_MISMATCH`: request.n_samples differs from expected value
  - `FIELD_MISMATCH`: request.valid_labels differs from expected value
  - `FIELD_MISMATCH`: request.model differs from expected value
  - `FIELD_MISMATCH`: request.group_name differs from expected value
  - `FIELD_MISMATCH`: request.transport differs from expected value
  - `FIELD_MISMATCH`: request.instructions differs from expected value
  - `HASH_MISSING_OR_INVALID`: request.input_sha256
  - `HASH_MISSING_OR_INVALID`: request.prompt_file_sha256
  - `FIELD_MISMATCH`: request.user_message differs from expected value
  - `HASH_MISSING_OR_INVALID`: request.full_prompt_sha256
- `claude-sonnet-4-6/tep_misnamed_r1`:
  - `HASH_MISMATCH`: metadata.prediction_sha256
- `claude-sonnet-4-6/tep_misnamed_r3`:
  - `REPORTED_CONTAMINATED`: Existing metadata explicitly marks this session CONTAMINATED; it cannot be used in accuracy comparisons. This audit does not infer contamination from identical outputs.
  - `REPORTED_SUSPECT`: Existing metadata explicitly marks data integrity as suspect.

## Snapshot meaning

The JSON report contains SHA-256 hashes of the current allowlisted input and session artifacts. These hashes describe the audit-time bytes only; they are not a pre-score freeze. No result, metadata, protocol, original v3 core file, or historical manifest was changed.

This audit intentionally does not compute accuracy or confidence intervals and never reads datasets.json. A new controlled inference run with preserved provider-native request, response, identity, and tool-isolation evidence would be a separate experiment; current artifacts cannot be retroactively made eligible.
