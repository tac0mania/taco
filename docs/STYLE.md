# Writing style: ASD-STE100

All documents in this repository use ASD-STE100 Simplified Technical English. This includes
`docs/`, `HANDOFF.md`, `CLAUDE.md`, ADRs, status files and commit message bodies. Code comments
use the same rules where possible.

## Rules

1. **Short sentences.** A procedural sentence (an instruction) has 20 words or fewer. A
   descriptive sentence has 25 words or fewer.
2. **One instruction in each sentence.** If two actions occur at the same time, put them in one
   sentence.
3. **Use the imperative for instructions.** Write "Run the test", not "The test should be run".
4. **Use the active voice.** Write "The decoder verifies the checksum", not "The checksum is
   verified by the decoder".
5. **Use only simple verb tenses:** the present, the simple past, and the future.
6. **One word, one meaning.** Use one term for one thing. Use the terms in the plan (§2) and do
   not use synonyms for them.
7. **Do not use the "-ing" form** as a noun or an adjective where you can avoid it. Write "the
   coder", not "the coding component".
8. **Use articles** ("a", "an", "the") and do not omit words to make text shorter.
9. **Short paragraphs.** A paragraph has one topic and six sentences or fewer.
10. **Put warnings first.** A warning or a caution comes before the instruction that it applies
    to.
11. **Numbers.** Write units. Label each number `measured`, `published` or `est.`.
12. **Lists.** Use a vertical list for steps or for items that you must compare.

## Examples

| Do not write | Write |
|---|---|
| It should be noted that the coder might potentially fail if tables mismatch. | The coder can fail if the tables are different. |
| Running the check before committing is recommended. | Run the check. Then commit. |
| The file was compressed and it was then sent to devbox where decoding occurred. | The Mac compressed the file. devbox decoded it. |
