# Sample documents for Doc Processing AI

Upload these from the Documents page (or drag into the dropzone) to exercise RAG/chat.

| File | Good test questions |
|------|---------------------|
| `acme-benefits.md` | What is the Acme Care Plus deductible? How much is the wellness stipend? |
| `project-nova-brief.txt` | What is Project Nova’s launch target? What are the success metrics? |
| `refund-shipping-policy.md` | How many days do customers have to request a refund? What is overnight shipping cost? |

Suggested flow:

1. Upload all three files and wait until status is `ready`
2. Open **Chat** → mode **RAG**
3. Ask one of the questions above; confirm citation chips appear
4. Switch to **Agent** and ask a multi-part question across policies

Automated checks: after these files are indexed (`ready`), run
`python -m evals.run_evals` from `apps/api` (see `apps/api/evals/`).
