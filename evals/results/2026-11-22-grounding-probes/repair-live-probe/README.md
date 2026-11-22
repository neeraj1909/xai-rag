# Grounding repair exploratory probe (2026-11-22)

A live `gpt-4o-mini` call using the proposed repair prompt, the historical E2E question and full original evidence generated these two cited claims:

1. RAG couples a retrieval system with a language model.
2. RAG allows the model to generate answers grounded in fetched evidence rather than relying solely on parametric memory.

The response reported 552 prompt tokens and 106 completion tokens. These claims were **not verified end to end** by this probe. A direct, cold two-claim `FaithfulnessChecker.check_claims` call failed with:

```text
File "src/xai_rag/explainability/faithfulness.py", line 140, in _check_single_claim
    scores = self._score_pair(premise=chunk.content, hypothesis=claim.text)
File "src/xai_rag/explainability/faithfulness.py", line 95, in _score_pair
    model, tokenizer = self._get_model_and_tokenizer()
File "src/xai_rag/explainability/faithfulness.py", line 88, in _get_model_and_tokenizer
    return _load_nli_model(self._model_name, self._revision)
File "src/xai_rag/explainability/faithfulness.py", line 53, in _load_nli_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
ImportError: cannot import name 'AutoModelForSequenceClassification' from 'transformers'
```

The checker dispatches claims to two threads; its cached loader does not promise single-flight initialization. A concurrent lazy-import race is a plausible explanation, not a confirmed root cause: this was one observation, not a reproduction. The actual repair follows a first NLI check, which would ordinarily warm the loader, but a cold original multi-claim request can still encounter this failure and fail closed. A subsequent separate pre-warmed attempt loaded weights but produced no verdict before process exit; the cause of exit is unknown. No further resource-heavy probing was performed because the host was under memory pressure.

Independent probes of the specific shorter claim `RAG combines retrieval and language models.` found the existing NLI and OpenAI judge both pass it on the same historical premise, but that is not the output of the generation call above and does not establish that this live generation passed both gates.
