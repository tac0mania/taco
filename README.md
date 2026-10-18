# Taco

*Hard shell. No floats. Same taco on every machine.*

Taco is a lossless compressor that uses a language model. HardShell, its engine, uses only
integer arithmetic. Thus a file that one machine folds unfolds on every other machine, byte for
byte.

```
taco fold notes.txt        # makes notes.txt.taco
taco unfold notes.txt.taco # restores notes.txt
```

Status: phase 0 (start). The commands do not work yet. See `HANDOFF.md`.

- Design: `docs/design/plan.md`
- Architecture: `docs/ARCHITECTURE.md`
- Working rules: `CLAUDE.md`
