# Web (`apps/web`)

React + TypeScript UI for document management and grounded chat.
Styled with **Material UI** and motion via **Framer Motion** (Ink & Teal theme).

## Setup

```bash
npm install
npm run dev
```

Dev server: http://localhost:5173  
API calls go to `/api/*` and are proxied to `http://localhost:8000`.

## Scripts

- `npm run dev` — Vite dev server
- `npm run build` — production build
- `npm run preview` — preview build

## UI stack

- `@mui/material` + `@mui/icons-material` — layout, tables, chips, inputs
- `framer-motion` — route transitions, list stagger, chat bubble enter
- Theme: `src/theme.ts` (Plus Jakarta Sans + Source Serif 4, teal accent)

## Features

- Upload PDF / TXT / MD (animated dropzone)
- Document list with indexing status chips
- Chat with RAG or Agent mode
- Streaming answers + citation chips

## Env / config

No frontend secret. Gemini key lives only on the API.  
Optional: change proxy target in `vite.config.ts`.

## Folder map

- `src/theme.ts` — MUI theme
- `src/api` — typed client + SSE parser
- `src/pages` — Documents, Chat
- `src/components` — AppShell, UploadDropzone, DocumentTable, ChatMessage, CitationList, PageTransition
- `src/types` — shared DTOs mirroring backend schemas
