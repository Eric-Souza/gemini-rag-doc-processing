import SendRoundedIcon from '@mui/icons-material/SendRounded'
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Paper,
  Snackbar,
  Stack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material'
import { useEffect, useMemo, useRef, useState } from 'react'
import { listDocuments, streamChat } from '../api/client'
import { ChatBubble } from '../components/ChatMessage'
import type { ChatMessage, ChatMode, Citation, DocumentItem } from '../types'

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

export function ChatPage() {
  const [mode, setMode] = useState<ChatMode>('rag')
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [selected, setSelected] = useState<DocumentItem[]>([])
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    void listDocuments()
      .then(setDocuments)
      .catch((e: Error) => setError(e.message))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const readyDocs = useMemo(
    () => documents.filter((d) => d.status === 'ready'),
    [documents],
  )

  const send = async () => {
    const text = input.trim()
    if (!text || streaming) return

    const userMsg: ChatMessage = { id: uid(), role: 'user', content: text }
    const assistantId = uid()
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      citations: [],
      streaming: true,
      status: 'Connecting…',
    }

    setMessages((prev) => [...prev, userMsg, assistantMsg])
    setInput('')
    setStreaming(true)
    setError(null)

    const citations: Citation[] = []
    let sawToken = false

    try {
      await streamChat(
        {
          message: text,
          mode,
          document_ids: selected.length ? selected.map((d) => d.id) : undefined,
        },
        {
          onStatus: (status) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, status } : m,
              ),
            )
          },
          onToken: (token) => {
            sawToken = true
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + token, status: undefined }
                  : m,
              ),
            )
          },
          onCitation: (c) => {
            citations.push(c)
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, citations: [...citations] } : m,
              ),
            )
          },
          onDone: () => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, streaming: false, status: undefined }
                  : m,
              ),
            )
          },
          onError: (message) => {
            setError(message)
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      streaming: false,
                      status: undefined,
                      content: m.content || `Error: ${message}`,
                    }
                  : m,
              ),
            )
          },
        },
      )
      if (!sawToken) {
        // ensure status cleared if stream ended without tokens
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, status: undefined } : m,
          ),
        )
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Chat failed'
      setError(msg)
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                streaming: false,
                status: undefined,
                content: m.content || `Error: ${msg}`,
              }
            : m,
        ),
      )
    } finally {
      setStreaming(false)
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, streaming: false, status: undefined }
            : m,
        ),
      )
    }
  }

  return (
    <Stack spacing={3} sx={{ height: 'calc(100vh - 140px)', minHeight: 480 }}>
      <Box>
        <Typography variant="h4" gutterBottom>
          Chat
        </Typography>
        <Typography color="text.secondary">
          Ask questions grounded in your uploaded documents. Switch to Agent for
          tool-calling retrieval via LangGraph.
        </Typography>
      </Box>

      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
        <ToggleButtonGroup
          exclusive
          value={mode}
          onChange={(_, v) => v && setMode(v)}
          size="small"
          color="primary"
        >
          <ToggleButton value="rag">RAG</ToggleButton>
          <ToggleButton value="agent">Agent</ToggleButton>
        </ToggleButtonGroup>

        <Autocomplete
          multiple
          options={readyDocs}
          getOptionLabel={(o) => o.filename}
          value={selected}
          onChange={(_, v) => setSelected(v)}
          sx={{ flex: 1 }}
          renderInput={(params) => (
            <TextField
              {...params}
              label="Limit to documents (optional)"
              placeholder="All ready docs"
              size="small"
            />
          )}
        />
      </Stack>

      <Paper
        sx={{
          flex: 1,
          p: 2,
          overflowY: 'auto',
          bgcolor: 'rgba(255,255,255,0.7)',
        }}
      >
        {!messages.length && (
          <Box sx={{ py: 8, textAlign: 'center' }}>
            <Typography color="text.secondary">
              {readyDocs.length
                ? 'Ask anything about your indexed documents.'
                : 'Index at least one document on the Documents page first.'}
            </Typography>
          </Box>
        )}
        {messages.map((m) => (
          <ChatBubble key={m.id} message={m} />
        ))}
        <div ref={bottomRef} />
      </Paper>

      <Stack direction="row" spacing={1.5} sx={{ alignItems: 'flex-end' }}>
        <TextField
          fullWidth
          multiline
          maxRows={4}
          placeholder="Ask a question…"
          value={input}
          disabled={streaming}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              void send()
            }
          }}
        />
        <Button
          variant="contained"
          size="large"
          endIcon={<SendRoundedIcon />}
          disabled={streaming || !input.trim()}
          onClick={() => void send()}
          sx={{ minWidth: 120, height: 56 }}
        >
          Send
        </Button>
      </Stack>

      <Snackbar
        open={Boolean(error)}
        autoHideDuration={6000}
        onClose={() => setError(null)}
      >
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>
    </Stack>
  )
}
