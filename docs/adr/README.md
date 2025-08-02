# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the Autoformalize Math Lab project.

## What are ADRs?

Architecture Decision Records (ADRs) are short text documents that capture an important architectural decision made along with its context and consequences.

## Format

We use the following format for ADRs:

```
# ADR-XXXX: [Title]

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
[What is the issue that we're seeing that is motivating this decision or change?]

## Decision
[What is the change that we're proposing and/or doing?]

## Consequences
[What becomes easier or more difficult to do because of this change?]
```

## Index of ADRs

- [ADR-0001: Choice of Programming Language](adr-0001-programming-language.md)
- [ADR-0002: LLM Provider Selection](adr-0002-llm-provider-selection.md)
- [ADR-0003: Self-Correction Strategy](adr-0003-self-correction-strategy.md)
- [ADR-0004: Proof Assistant Interface Design](adr-0004-proof-assistant-interface.md)
- [ADR-0005: Caching Strategy](adr-0005-caching-strategy.md)

## Creating New ADRs

1. Copy the template from `adr-template.md`
2. Use the next available number in the sequence
3. Use a descriptive title in kebab-case
4. Update this README to include the new ADR in the index