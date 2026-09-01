# Training inputs

R4 (Round 4) is ID-only organizer-train contrastive supervision. R5 (Round 5)
is the fully synthetic OpenRouter-generated corpus. R7 (Round 7) is
query-specific ID-only organizer-train relation supervision used only by the
candidate branch.

| Channel | Volume | Use |
|---|---:|---|
| Organizer-train FLV CE | 5,502 rows, 198 positives | control and candidate LoRA |
| R4 ID-only real-real relations | 100 pairs | pairwise loss in both branches |
| R5 OpenRouter synthetic pairs | 200 pairs / 400 cards | pairwise loss in both branches |
| R7 ID-only real relations | 100 query instances / 62 q4-full-fit mass keys | candidate-only pairwise loss |

R5 is never added to CE. Structural and direction checks pass for 200/200
pairs; 192/200 pass usable review and all eight reviewer flags are retained.
