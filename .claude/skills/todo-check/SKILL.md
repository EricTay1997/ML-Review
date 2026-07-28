---
name: todo-check
description: Use whenever committing or pushing changes in this repo (git commit, git push, "push this", "commit this"). Cross-checks llms/TODO.md against the changes being shipped and ticks off write-up-backlog items whose notes have now been incorporated.
---

# TODO cross-check on push

Before (or as part of) any commit/push in this repo:

1. Look at the changes being shipped: `git diff --stat @{push}..HEAD` plus anything staged (`git diff --cached --stat`). Focus on files under `llms/`.
2. Open `llms/TODO.md`. For each **unchecked** item (`- [ ]`), decide whether the shipped changes actually incorporate that source into its target file — i.e. real note content was added, not just a link or a stub edit.
3. For each item that is now incorporated, mark it `- [x]` in `llms/TODO.md` and include that edit in the same commit/push.
4. Do NOT check off items whose target file was merely touched, reorganized, or linked — content from the source must have landed.
5. If new reading links appear in the shipped notes that aren't on the list and aren't yet incorporated, ask whether to add them to `llms/TODO.md` under the right folder heading.
6. In your summary, report: items checked off this push (if any) and the count remaining unchecked.
