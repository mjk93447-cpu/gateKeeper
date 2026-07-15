# Documentation and text standard

All repository documentation, operator manuals, generated release notes,
configuration descriptions, UI menu labels, and plugin documentation must be
written in clear English and saved as UTF-8.

Requirements:

- use English for README files, Markdown, YAML comments, JSON descriptions, and
  release notes;
- use ASCII punctuation where practical and avoid corrupted or locale-dependent
  text encodings;
- keep code identifiers and log event names in English;
- include an English operator manual and PLC integration manual in every release;
- run `python scripts/validate_english_docs.py` before a commit or release;
- translations may be added as separate files, but the English file remains the
  canonical specification.
