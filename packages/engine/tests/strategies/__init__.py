"""Reusable, bounded Hypothesis generators for the property and metamorphic tests.

Every generator keeps degree, coefficient size and dimension small on purpose: an example must cost milliseconds
(or one sandbox call), so the pr profile (8 derandomized examples) stays cheap and the extended profile can run
hundreds. Hypothesis shrinks a failure to a minimal object; pin it with `@example` so the pr tier keeps it.
"""
