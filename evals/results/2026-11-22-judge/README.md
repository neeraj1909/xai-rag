# Fictional judge sensitivity, 22 November 2026

The [active runner](../../judge/README.md) evaluated the same 16 fictional,
author-labelled candidate answers and rubric as the earlier
[historical study](../../archive/interview-evidence/judge-fixture-study.md).
The case fixture SHA-256 is
`ad995dbfba6239085cf65b1ceec8637197de74bc89beb259e470d9eb06aa0626`.
These labels are not independent human review, and the cases are not a
representative benchmark or an assessment of the actual XAI-RAG application.

The [Codex CLI run](codex-judgments.json) requested `gpt-5.6-luna` in 16
separate ephemeral read-only agent sessions. All 16 returned schema-valid
judgments. Agreement with authored faithfulness expectations was 16/16,
including two no-claim cases expected to receive null (14/14 excluding those
cases). Correctness agreement was 16/16. The recorded aggregate CLI usage was
341,388 input and 1,674 output tokens. Inspect the per-case reasons and labels
in the report; 16 cases do not establish performance on unseen examples.
The locally installed CLI reported `codex-cli 0.171.0`; the runner used Python
3.14.2.

This CLI run used agent instructions and an output schema, not the original
OpenAI chat-completions interface. The CLI did not expose a returned model
revision, or the temperature and maximum output token controls used by the old
API experiment. These different conditions make the scores an exploratory
comparison, not a controlled reproduction or paired benchmark.

For the Codex run, the runner SHA-256 was
`f18e31c86d79c4cb98f0d8d26f9c7a9e8a4977e62f96fb716d2be721bb45929c`
and the Codex backend SHA-256 was
`bf95bbf5fe5e5f0308c88d5564404986514ae400a41172aed4a39999329fdb25`.
The judgment report SHA-256 is
`4e1e1bb70b85c61dd529388c76caa5f2f0d93dc56c027118891eba0622cc9af3`.
The exact two runner files are retained in `measured-runner.tar.gz`, SHA-256
`f77f996927bf0f17505aa846d1c918b9495bc10b3b4cec43f6902d63910aa7f5`;
this archive is not a complete environment snapshot.

A separate [OpenAI API attempt](openai-auth-failure.json), requesting
`gpt-4.1-mini`, returned HTTP 401 authentication errors on all 16 attempts;
0/16 cases were judged. A separate one-case check with the existing standard
`OPENAI_API_KEY` also returned HTTP 401. Neither attempt established an API
model score or demonstrated an API-credit or compute limitation. No credential
values are recorded.
