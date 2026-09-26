"""User-facing guardrail messages (single source of truth)."""

DISPOSITION_REFUSAL = (
    "**I can't approve, reject, release or otherwise disposition a batch.** I can provide investigation support - "
    "the evidence, historical comparison and the procedures QA should follow - but final disposition requires "
    "authorized Quality personnel. Their decision belongs in your quality system (QMS). "
    "You can record a review of this AI finding on the **QA Review** page.")
INJECTION_REFUSAL = (
    "**This request was blocked.** It appears to ask the assistant to ignore or change its rules, reveal internal "
    "instructions or credentials. I can only investigate batches using the batch data and quality procedures. "
    "Please rephrase your question about the batch.")
DISCLAIMER = ("_Decision support only. This is not a batch disposition; final approval, rejection or release rests "
              "with authorized Quality personnel._")
GENERIC_ERROR = ("Something went wrong while processing your request. No batch was changed. "
                 "Please try again; if it persists, contact support with reference **{ref}**.")
OUTPUT_REPLACED = ("The AI-written answer failed automatic safety validation and was replaced by a deterministic "
                   "summary built directly from the calculated analysis.")
